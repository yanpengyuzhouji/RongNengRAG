import assert from 'node:assert/strict'

globalThis.sessionStorage = {
  getItem() { return '' },
  setItem() {},
  removeItem() {},
}

let requestBody
globalThis.fetch = async (_url, options) => {
  requestBody = JSON.parse(options.body)
  return new Response('', { status: 200 })
}

const { askWorkbookStream } = await import('../src/api.js')
const { consumeStream, validateSseResponse } = await import('../src/sse.js')

await askWorkbookStream('book', 'question', 200, 'conv-123').promise
assert.equal(requestBody.conv_id, 'conv-123')

const payload = 'data: {"type":"done","done":true,"full_answer":"ok","conv_id":"conv-456"}\n\n'
const result = await new Promise((resolve, reject) => {
  consumeStream(new Response(payload), { onDone: resolve, onError: reject })
})
assert.equal(result.convId, 'conv-456')

const eofError = await new Promise((resolve) => {
  consumeStream(
    new Response('data: {"type":"status","status":"thinking"}\n\n', {
      headers: { 'Content-Type': 'text/event-stream' },
    }),
    { onError: resolve }
  )
})
assert.match(eofError.message, /before a done event/)

await assert.rejects(
  validateSseResponse(new Response('{"detail":"bad request"}', {
    status: 400,
    headers: { 'Content-Type': 'application/json' },
  })),
  /bad request/
)

await assert.rejects(
  validateSseResponse(new Response('{"error":"not an event stream"}', {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })),
  /not an event stream/
)

let errorCount = 0
const typedError = await new Promise((resolve) => {
  consumeStream(
    new Response(
      'data: {"type":"error","error":"boom"}\n\n' +
      'data: {"type":"done","done":true}\n\n'
    ),
    {
      onError(error) {
        errorCount += 1
        resolve(error)
      },
    }
  )
})
assert.equal(typedError.message, 'boom')
assert.equal(errorCount, 1)

let streamCancelled = false
const pendingResponse = new Response(new ReadableStream({
  cancel() { streamCancelled = true },
}))
const pendingConsumer = consumeStream(pendingResponse, {
  onError() { throw new Error('cancel must not be reported as an error') },
})
pendingConsumer.cancel()
await pendingConsumer.finished
assert.equal(streamCancelled, true)
