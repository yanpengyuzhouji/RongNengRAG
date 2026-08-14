/**
 * SSE stream consumer via ReadableStream.
 *
 * Two contracts, fully compatible:
 * - Legacy (/ask/stream): `data.token` -> onToken, `data.done` -> onDone.
 * - Typed (Excel /excel/.../ask): every event carries a `type` field
 *   (status / sql / table / html / plan.update / token / done / error).
 *   Structured events are dispatched to `onEvent(event)`; legacy token/done
 *   handling still works for AIAssistant.vue.
 *
 * Returns { cancel() } to abort mid-stream.
 */
export function consumeStream(
  response,
  { onToken, onDone, onError, onEvent } = {}
) {
  const reader = response.body?.getReader()
  if (!reader) {
    onError?.(new Error('Response body is not readable'))
    return { cancel() {} }
  }

  const decoder = new TextDecoder()
  let buffer = ''
  let cancelled = false
  let settled = false

  function fail(error) {
    if (settled || cancelled) return
    settled = true
    onError?.(error instanceof Error ? error : new Error(String(error)))
  }

  function finish(result) {
    if (settled || cancelled) return
    settled = true
    onDone?.(result)
  }

  function processLine(line) {
    if (!line.startsWith('data:')) return false
    const payload = line.slice(5).trimStart()
    if (!payload) return false
    let data
    try {
      data = JSON.parse(payload)
    } catch {
      return false
    }

    // ── Typed events (Excel ReAct) ──
    if (data.type) {
      if (data.type === 'error') {
        if (data.conv_id) onEvent?.(data)
        fail(new Error(data.error || 'stream error'))
        return true
      }
      if (data.type === 'done') {
        onEvent?.(data)
        finish({
          fullAnswer: data.full_answer || '',
          citations: data.citations || [],
          sources: data.sources || [],
          reportUrl: data.report_url || '',
          reportTitle: data.report_title || '',
          elapsedMs: data.elapsed_ms || 0,
          convId: data.conv_id || '',
        })
        return true
      }
      onEvent?.(data)
      if (data.type === 'status' && data.status === 'thinking') {
        onToken?.('__status__thinking')
      }
      return false
    }

    // ── Legacy /ask/stream contract ──
    if (data.error) {
      fail(new Error(data.error))
      return true
    }
    if (data.token && !data.done) onToken?.(data.token)
    if (data.done) {
      finish({
        fullAnswer: data.full_answer || '',
        citations: data.citations || [],
        sources: data.sources || [],
        elapsedMs: data.elapsed_ms || 0,
        convId: data.conv_id || '',
      })
      return true
    }
    if (data.status === 'searching') onToken?.('__status__searching')
    if (data.status === 'thinking') onToken?.('__status__thinking')
    return false
  }

  async function read() {
    try {
      while (!cancelled) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (processLine(line) || settled) return
        }
      }
      buffer += decoder.decode()
      if (buffer && processLine(buffer)) return
      if (!cancelled && !settled) {
        fail(new Error('SSE stream ended before a done event'))
      }
    } catch (e) {
      fail(e)
    }
  }

  const finished = read()

  return {
    finished,
    cancel() {
      cancelled = true
      settled = true
      reader.cancel()
    },
  }
}

export async function validateSseResponse(response) {
  const contentType = (response.headers.get('content-type') || '').toLowerCase()
  if (!response.ok || !contentType.includes('text/event-stream')) {
    const text = await response.text().catch(() => '')
    let detail = text
    try {
      const parsed = JSON.parse(text)
      detail = parsed.detail || parsed.error || text
    } catch {
      // keep text body
    }
    const reason = detail ? `: ${String(detail).slice(0, 300)}` : ''
    throw new Error(`Excel SSE request failed (HTTP ${response.status})${reason}`)
  }
  return response
}
