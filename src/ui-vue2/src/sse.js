/**
 * SSE stream consumer via ReadableStream.
 * Calls onToken(token) for each content token, onDone(result) when finished.
 * Returns { cancel() } to abort mid-stream.
 */
export function consumeStream(response, { onToken, onDone, onError }) {
  const reader = response.body?.getReader()
  if (!reader) {
    onError?.(new Error('Response body is not readable'))
    return { cancel() {} }
  }

  const decoder = new TextDecoder()
  let buffer = ''
  let cancelled = false

  async function read() {
    try {
      while (!cancelled) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.slice(6))
            if (data.token && !data.done) {
              onToken?.(data.token)
            }
            if (data.done) {
              onDone?.({
                fullAnswer: data.full_answer || '',
                citations: data.citations || [],
                sources: data.sources || [],
                elapsedMs: data.elapsed_ms || 0,
              })
              return
            }
            if (data.status === 'searching') {
              onToken?.('__status__searching')
            }
            if (data.status === 'thinking') {
              onToken?.('__status__thinking')
            }
          } catch {
            // skip unparseable lines
          }
        }
      }
    } catch (e) {
      if (!cancelled) onError?.(e)
    }
  }

  read()

  return {
    cancel() {
      cancelled = true
      reader.cancel()
    },
  }
}
