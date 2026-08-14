// 后端 API 地址: 优先读环境变量；未配置时跟随当前页面主机。
// 这样局域网设备访问前端时不会把 localhost 误指向客户端自身。
// 部署时可通过 .env 或命令行配置: VITE_API_BASE=http://192.168.x.x:8000 npm run dev
const pageHost = typeof window !== 'undefined' ? window.location.hostname : 'localhost'
const pageProtocol = typeof window !== 'undefined' ? window.location.protocol : 'http:'
const BASE = (import.meta.env?.VITE_API_BASE || `${pageProtocol}//${pageHost}:8000`).replace(/\/$/, '')

function authHeaders() {
  const key = sessionStorage.getItem('rongneng_api_key') || import.meta.env?.VITE_API_KEY || ''
  return key ? { 'X-API-Key': key } : {}
}

export function setApiKey(key) {
  if (key) sessionStorage.setItem('rongneng_api_key', key)
  else sessionStorage.removeItem('rongneng_api_key')
}

export class ApiError extends Error {
  constructor(message, { status = null, body = null, url = '', code = 'API_ERROR', cause } = {}) {
    super(message, cause ? { cause } : undefined)
    this.name = 'ApiError'
    this.status = status
    this.body = body
    this.url = url
    this.code = code
  }
}

async function responseBody(resp) {
  const text = await resp.text().catch(() => '')
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

function bodyMessage(body) {
  if (typeof body === 'string') return body.slice(0, 200)
  if (body && typeof body.detail === 'string') return body.detail.slice(0, 200)
  if (body && typeof body.message === 'string') return body.message.slice(0, 200)
  return ''
}

/**
 * Unified fetch wrapper with error handling.
 * @param {string} method
 * @param {string} path - API path like '/files/summary'
 * @param {object} opts - { body, params, timeout, isFormData }
 * @returns {Promise<object|string|null>} parsed response; failures reject with ApiError
 */
export async function api(method, path, opts = {}) {
  const {
    body,
    params,
    timeout = 120000,
    isFormData,
    throwOnError = path.startsWith('/excel/'),
  } = opts
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

  const headers = authHeaders()
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
    if (!resp.ok) {
      const responsePayload = await responseBody(resp)
      const detail = bodyMessage(responsePayload)
      throw new ApiError(
        `API 请求失败 (HTTP ${resp.status})${detail ? `: ${detail}` : ''}`,
        {
          status: resp.status,
          body: responsePayload,
          url,
          code: 'HTTP_ERROR',
        },
      )
    }

    return responseBody(resp)
  } catch (e) {
    let error = e
    if (e.name === 'AbortError') {
      error = new ApiError(`请求超时 (${timeout / 1000}s)`, {
        url,
        code: 'TIMEOUT',
        cause: e,
      })
    } else if (!(e instanceof ApiError)) {
      error = new ApiError(`无法连接后端: ${e?.message || e}`, {
        url,
        code: 'NETWORK_ERROR',
        cause: e,
      })
    }
    if (throwOnError) throw error
    return `[API Error] ${error.message}`
  } finally {
    clearTimeout(timer)
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

export function getFileContent(identifier, editable = false) {
  return api('GET', `/files/${encodeURIComponent(identifier)}/content`, {
    params: { editable: editable ? 'true' : undefined },
    throwOnError: true,
  })
}

export function saveFileContent(identifier, baseRevision, edits) {
  return api('PUT', `/files/${encodeURIComponent(identifier)}/content`, {
    body: { base_revision: baseRevision, edits },
    timeout: 600000,
    throwOnError: true,
  })
}

export function compareFileOcr(identifier) {
  return api('GET', `/files/${encodeURIComponent(identifier)}/ocr-compare`, {
    timeout: 600000,
    throwOnError: true,
  })
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
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
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

function downloadNameFromResponse(response, fallback = '') {
  const disposition = response.headers.get('content-disposition') || ''
  const encoded = disposition.match(/filename\*\s*=\s*(?:UTF-8''|utf-8'')?([^;]+)/i)
  if (encoded?.[1]) {
    try { return decodeURIComponent(encoded[1].trim().replace(/^"|"$/g, '')) } catch { /* fallback below */ }
  }
  const quoted = disposition.match(/filename\s*=\s*"([^"]+)"/i)
  if (quoted?.[1]) return quoted[1]
  const plain = disposition.match(/filename\s*=\s*([^;]+)/i)
  return plain?.[1]?.trim().replace(/^"|"$/g, '') || fallback || 'download'
}

export async function downloadFile(identifier, fallbackName = '') {
  const url = `${BASE}/files/${encodeURIComponent(identifier)}/download`
  const response = await fetch(url, { headers: authHeaders() })
  if (!response.ok) {
    const payload = await responseBody(response)
    throw new ApiError(
      `文件下载失败 (HTTP ${response.status})${bodyMessage(payload) ? `: ${bodyMessage(payload)}` : ''}`,
      { status: response.status, body: payload, url, code: 'HTTP_ERROR' },
    )
  }
  const objectUrl = URL.createObjectURL(await response.blob())
  try {
    const link = document.createElement('a')
    link.href = objectUrl
    link.download = downloadNameFromResponse(response, fallbackName || identifier)
    link.rel = 'noopener'
    link.click()
  } finally {
    URL.revokeObjectURL(objectUrl)
  }
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
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
  })
}

// ═══ Excel 工作簿 (经 /excel 代理 → excel-workbook-service) ═══

export function createWorkbookImport(file, sourceChannel = 'agent') {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('source_channel', sourceChannel)
  return api('POST', '/excel/workbook-imports', {
    body: fd,
    isFormData: true,
    timeout: 600000,
  })
}

export function listWorkbookImports(options = {}) {
  const { status, limit, offset } = options
  const params = { limit: limit ?? 200, offset: offset ?? 0 }
  if (status) params.status = status
  return api('GET', '/excel/workbook-imports', { params })
}

export function getWorkbookImport(importId) {
  return api('GET', `/excel/workbook-imports/${importId}`)
}

export function getSheetRows(importId, sheetId, offset = 0, limit = 100) {
  return api('GET', `/excel/workbook-imports/${importId}/sheets/${sheetId}/rows`, {
    params: { offset, limit },
  })
}

export function updateDraft(importId, payload) {
  return api('PATCH', `/excel/workbook-imports/${importId}/draft`, { body: payload })
}

export function validateImport(importId) {
  return api('POST', `/excel/workbook-imports/${importId}/validate`)
}

export function confirmImport(importId, payload) {
  return api('POST', `/excel/workbook-imports/${importId}/confirm`, { body: payload })
}

export function queryWorkbook(workbookId, sql, maxRows = 200) {
  return api('POST', `/excel/workbooks/${workbookId}/query`, {
    body: { sql, max_rows: maxRows },
  })
}

/** Returns { promise, controller } for SSE consumption */
export function askWorkbookStream(workbookId, question, maxRows = 200, convId = null) {
  const controller = new AbortController()
  const promise = fetch(`${BASE}/excel/workbooks/${workbookId}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({
      question,
      max_rows: maxRows,
      ...(convId ? { conv_id: convId } : {}),
    }),
    signal: controller.signal,
  })
  return { promise, controller }
}

export function workbookReportUrl(workbookId, reportId) {
  return `${BASE}/excel/workbooks/${workbookId}/reports/${reportId}`
}

export async function fetchWorkbookReport(workbookId, reportId, signal) {
  const response = await fetch(workbookReportUrl(workbookId, reportId), {
    headers: authHeaders(),
    signal,
  })
  if (!response.ok) {
    const detail = await response.text().catch(() => '')
    throw new Error(`报告加载失败 (HTTP ${response.status}): ${detail.slice(0, 200)}`)
  }
  return response.blob()
}
