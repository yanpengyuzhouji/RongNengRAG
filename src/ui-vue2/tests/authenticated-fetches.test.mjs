import assert from 'node:assert/strict'

globalThis.sessionStorage = {
  getItem: (name) => name === 'rongneng_api_key' ? 'test-secret' : '',
  setItem: () => {},
  removeItem: () => {},
}

const clicks = []
globalThis.document = {
  createElement: () => ({
    click() { clicks.push(this) },
  }),
}
globalThis.URL.createObjectURL = () => 'blob:test'
globalThis.URL.revokeObjectURL = () => {}

const requests = []
globalThis.fetch = async (url, options = {}) => {
  requests.push({ url, options })
  if (url.includes('/download')) return new Response('file')
  return new Response(null, { status: 204 })
}

const { askStream, downloadFile, recoverPending } = await import('../src/api.js')

await askStream('question').promise
await recoverPending()
await downloadFile('hash')

assert.equal(requests.length, 3)
for (const request of requests) {
  assert.equal(request.options.headers['X-API-Key'], 'test-secret')
}
assert.equal(clicks.length, 1)
assert.equal(clicks[0].download, 'hash')
