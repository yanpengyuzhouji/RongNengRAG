import assert from 'node:assert/strict'

globalThis.sessionStorage = {
  getItem: () => '',
  setItem: () => {},
  removeItem: () => {},
}

const { api, ApiError } = await import('../src/api.js')

globalThis.fetch = async () => new Response(
  JSON.stringify({ detail: 'revision conflict' }),
  { status: 409, headers: { 'content-type': 'application/json' } },
)
await assert.rejects(
  () => api('PATCH', '/excel/test'),
  (error) => {
    assert.ok(error instanceof ApiError)
    assert.equal(error.status, 409)
    assert.equal(error.code, 'HTTP_ERROR')
    assert.deepEqual(error.body, { detail: 'revision conflict' })
    assert.match(error.message, /revision conflict/)
    return true
  },
)

globalThis.fetch = async () => new Response('upstream unavailable', { status: 502 })
await assert.rejects(
  () => api('GET', '/excel/test'),
  (error) => error.status === 502 && error.body === 'upstream unavailable',
)

globalThis.fetch = async () => new Response(null, { status: 204 })
assert.equal(await api('DELETE', '/excel/test'), null)

globalThis.fetch = async (_url, options) => new Promise((_resolve, reject) => {
  options.signal.addEventListener('abort', () => {
    reject(new DOMException('aborted', 'AbortError'))
  })
})
await assert.rejects(
  () => api('GET', '/excel/slow', { timeout: 5 }),
  (error) => error instanceof ApiError && error.code === 'TIMEOUT' && error.status === null,
)

globalThis.fetch = async () => { throw new TypeError('Failed to fetch') }
await assert.rejects(
  () => api('GET', '/excel/offline'),
  (error) => error instanceof ApiError && error.code === 'NETWORK_ERROR',
)

// Existing non-Excel callers retain their documented string-error contract.
assert.match(await api('GET', '/files'), /^\[API Error\]/)
