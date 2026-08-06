// 后端 API 地址: 优先读环境变量 VITE_API_BASE, 默认本机 8000
// 部署时可通过 .env 或命令行配置: VITE_API_BASE=http://192.168.x.x:8000 npm run dev
const BASE = (import.meta.env?.VITE_API_BASE || 'http://localhost:8000').replace(/\/$/, '')

/**
 * Unified fetch wrapper with error handling.
 * @param {string} method
 * @param {string} path - API path like '/files/summary'
 * @param {object} opts - { body, params, timeout, isFormData }
 * @returns {Promise<object|string>} - parsed JSON or error string
 */
export async function api(method, path, opts = {}) {
  const { body, params, timeout = 120000, isFormData } = opts
  let url = `${BASE}${path}`

  if (params) {
    // Filter out null/undefined/empty-string values to avoid serializing "undefined"
    const clean = {}
    for (const [k, v] of Object.entries(params)) {
      if (v !== null && v !== undefined && v !== '') clean[k] = v
    }
    if (Object.keys(clean).length) {
      const qs = new URLSearchParams(clean).toString()
      url += `?${qs}`
    }
  }

  const headers = {}
  if (!isFormData) {
    headers['Content-Type'] = 'application/json'
  }

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeout)

  try {
    const resp = await fetch(url, {
      method,
      headers,
      body: isFormData ? body : (body ? JSON.stringify(body) : undefined),
      signal: controller.signal,
    })
    clearTimeout(timer)

    if (!resp.ok) {
      const text = await resp.text().catch(() => '')
      return `[API Error] ${resp.status}: ${text.slice(0, 200)}`
    }

    const data = await resp.json().catch(() => null)
    if (data === null) return `[API Error] 返回空响应 (HTTP ${resp.status})`
    return data
  } catch (e) {
    clearTimeout(timer)
    if (e.name === 'AbortError') return `[API Error] 请求超时 (${timeout / 1000}s)`
    if (e.message?.includes('Failed to fetch') || e.message?.includes('NetworkError'))
      return '[API Error] 无法连接后端，请先启动 API 服务 (端口 8000)'
    return `[API Error] 请求失败: ${e.message}`
  }
}

// ── File APIs ──

export function uploadFile(file, domain, category) {
  const fd = new FormData()
  fd.append('file', file)
  if (domain && domain !== 'auto') fd.append('domain', domain)
  if (category) fd.append('category', category)
  return api('POST', '/upload', { body: fd, isFormData: true, timeout: 600000 })
}

export function listFiles(options = {}) {
  const { status, domain, limit, offset, includeDeleted } = options
  const params = { limit: limit ?? 500, offset: offset ?? 0, include_deleted: includeDeleted ?? false }
  if (status) params.status = status
  if (domain) params.domain = domain
  return api('GET', '/files', { params })
}

export function getFileSummary() {
  return api('GET', '/files/summary')
}

export function getFileDetail(identifier) {
  return api('GET', `/files/${encodeURIComponent(identifier)}`)
}

export function deleteFile(identifier, removeFile = true) {
  return api('DELETE', `/files/${encodeURIComponent(identifier)}`, {
    params: { remove_file: removeFile },
  })
}

export function syncFiles(dryRun = false, checkMilvus = false) {
  return api('POST', '/files/sync', {
    params: { dry_run: dryRun, check_milvus: checkMilvus },
    timeout: 300000,
  })
}

export function getStats() {
  return api('GET', '/stats')
}

// ── Chat / Search APIs ──

export function search(query, topK = 15, domainFilter = null) {
  return api('POST', '/search', {
    body: { query, top_k: topK, domain_filter: domainFilter },
    timeout: 60000,
  })
}

export function ask(query, topK = 15, domainFilter = null, conversationId = null) {
  return api('POST', '/ask', {
    body: { query, top_k: topK, domain_filter: domainFilter, conversation_id: conversationId },
    timeout: 300000,
  })
}

/** Returns the raw fetch Response for SSE consumption */
export function askStream(query, topK = 15, domainFilter = null, conversationId = null) {
  const controller = new AbortController()
  const promise = fetch(`${BASE}/ask/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      top_k: topK,
      domain_filter: domainFilter,
      conversation_id: conversationId,
    }),
    signal: controller.signal,
  })
  return { promise, controller }
}

// ── Conversation APIs ──

export function createConversation(title = '') {
  return api('POST', '/conversations', { body: { title } })
}

export function listConversations() {
  return api('GET', '/conversations')
}

export function getConversation(convId) {
  return api('GET', `/conversations/${convId}`)
}

export function deleteConversation(convId) {
  return api('DELETE', `/conversations/${convId}`)
}

// ── File management extras ──

export function updateFileMeta(identifier, fields) {
  return api('PATCH', `/files/${encodeURIComponent(identifier)}`, { body: fields })
}

export function downloadFile(identifier) {
  window.open(`${BASE}/files/${encodeURIComponent(identifier)}/download`, '_blank')
}

export function getSubcategories(domain, category) {
  const params = {}
  if (domain) params.domain = domain
  if (category) params.category = category
  return api('GET', '/files/subcategories', { params })
}

// ── Recover ──

/** Returns the raw fetch Response for SSE consumption */
export function recoverPending() {
  return fetch(`${BASE}/files/recover-pending`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  })
}
