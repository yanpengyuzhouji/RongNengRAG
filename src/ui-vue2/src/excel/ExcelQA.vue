<template>
  <div class="excel-qa">
    <el-card shadow="never" class="qa-card">
      <div class="qa-header">
        <span class="wb-name">📊 {{ workbook.display_name || '工作簿' }}</span>
        <div class="qa-session-bar">
          <el-select
            v-if="conversations.length"
            v-model="convId"
            size="small"
            class="conversation-select"
            :loading="historyLoading"
            placeholder="选择历史会话"
            @change="switchConversation"
          >
            <el-option
              v-for="conversation in conversations"
              :key="conversation.conv_id"
              :label="conversation.title || '未命名会话'"
              :value="conversation.conv_id"
            />
          </el-select>
          <el-button size="small" text :disabled="status === 'thinking'" @click="startNewConversation">
            新会话
          </el-button>
          <el-tag size="small" type="success">已建库 v{{ workbook.active_version }}</el-tag>
        </div>
      </div>

      <div ref="msgArea" class="msg-area">
        <div v-if="!messages.length" class="welcome">
          <p>对已入库的工作簿提问,可生成 SQL 查询与 HTML 分析报告。</p>
          <p class="hint">示例:「按门店汇总销量,生成 HTML 报告」</p>
        </div>

        <div
          v-for="(msg, i) in messages"
          :key="i"
          :class="['msg', msg.role === 'user' ? 'msg-user' : 'msg-assistant']"
        >
          <div v-if="msg.role === 'user'" class="msg-text">{{ msg.text }}</div>
          <template v-else>
            <div v-if="msg.text" class="msg-text">{{ msg.text }}</div>

            <div v-for="(t, ti) in msg.tables || []" :key="'t' + ti" class="table-block">
              <div class="table-caption">SQL 查询结果 ({{ t.row_count }} 行)</div>
              <el-table :data="t.rows" size="small" border max-height="300">
                <el-table-column
                  v-for="c in t.columns"
                  :key="c"
                  :prop="c"
                  :label="c"
                  :min-width="100"
                />
              </el-table>
            </div>

            <div v-for="(r, ri) in msg.reports || []" :key="'r' + ri">
              <HtmlReportViewer :workbook-id="workbook.workbook_id" :report-id="r.reportId" :title="r.title" />
            </div>
          </template>
        </div>

        <div v-if="status === 'thinking'" class="status-line">💭 思考中…</div>
        <div v-if="error" class="error-line">⚠ {{ error }}</div>
      </div>

      <div class="input-bar">
        <el-input
          v-model="input"
          placeholder="输入问题,如:按门店汇总销量并生成 HTML 报告"
          :disabled="status === 'thinking'"
          @keyup.enter="send"
        >
          <template #append>
            <el-button v-if="status === 'thinking'" @click="cancelRequest">
              取消
            </el-button>
            <el-button v-else type="primary" @click="send">
              发送
            </el-button>
          </template>
        </el-input>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import {
  askWorkbookStream,
  getWorkbookConversation,
  listWorkbookConversations,
} from '../api.js'
import { consumeStream, validateSseResponse } from '../sse.js'
import HtmlReportViewer from './HtmlReportViewer.vue'

const props = defineProps({
  workbook: { type: Object, required: true },
})

const input = ref('')
const messages = ref([])
const status = ref('idle')
const error = ref('')
const msgArea = ref(null)
const convId = ref(null)
const conversations = ref([])
const historyLoading = ref(false)
let activeConsumer = null
let activeController = null
let historyRequestSerial = 0

function conversationStorageKey() {
  return `excel-active-conversation:${props.workbook.workbook_id}`
}

function rememberConversation(id) {
  convId.value = id || null
  if (typeof window === 'undefined' || !window.localStorage) return
  if (id) window.localStorage.setItem(conversationStorageKey(), id)
  else window.localStorage.removeItem(conversationStorageKey())
}

function savedConversationId() {
  if (typeof window === 'undefined' || !window.localStorage) return null
  return window.localStorage.getItem(conversationStorageKey())
}

function reportIdFromUrl(url) {
  const value = String(url || '')
  const name = value.split('/').pop() || ''
  const raw = name.replace(/\.html(?:\?.*)?$/, '')
  try { return decodeURIComponent(raw) } catch { return raw }
}

function appendTable(assistant, table) {
  if (!table || !Array.isArray(table.columns) || !Array.isArray(table.rows)) return
  assistant.tables.push({
    columns: table.columns,
    rows: table.rows.map((row, index) => {
      const obj = {}
      table.columns.forEach((column, columnIndex) => { obj[column] = row[columnIndex] })
      obj.__row__ = index + 1
      return obj
    }),
    row_count: table.row_count || 0,
  })
}

function appendReport(assistant, report) {
  if (!report) return
  const reportId = report.report_id || report.reportId || reportIdFromUrl(report.url)
  if (!reportId || assistant.reports.some((item) => item.reportId === reportId)) return
  assistant.reports.push({
    reportId,
    title: report.title || '分析报告',
  })
}

function restoreMessages(rawMessages) {
  const turns = new Map()
  for (const message of rawMessages || []) {
    const round = Number(message.round_index || 0)
    if (!turns.has(round)) {
      turns.set(round, {
        round,
        user: null,
        assistant: { role: 'assistant', text: '', tables: [], reports: [] },
      })
    }
    const turn = turns.get(round)
    const role = message.role
    if (role === 'user') {
      turn.user = { role: 'user', text: message.content || '' }
    } else if (role === 'ai') {
      turn.assistant.text = message.content || ''
    } else if (role === 'tool') {
      const meta = message.meta || {}
      const detail = meta.detail || {}
      if (detail.table) appendTable(turn.assistant, detail.table)
      if (detail.html) appendReport(turn.assistant, detail.html)
    } else if (role === 'view') {
      try {
        const view = JSON.parse(message.content || '{}')
        if (view.type === 'html') appendReport(turn.assistant, view)
      } catch {
        // A malformed/legacy view message must not prevent the rest of the
        // conversation from being restored.
      }
    }
  }

  const restored = []
  for (const turn of [...turns.values()].sort((a, b) => a.round - b.round)) {
    if (turn.user) restored.push(turn.user)
    const assistant = turn.assistant
    if (assistant.text || assistant.tables.length || assistant.reports.length) {
      restored.push(assistant)
    }
  }
  messages.value = restored
  scrollBottom()
}

async function loadConversation(id) {
  if (!id) {
    messages.value = []
    return
  }
  historyLoading.value = true
  const serial = ++historyRequestSerial
  try {
    const detail = await getWorkbookConversation(props.workbook.workbook_id, id)
    if (serial !== historyRequestSerial) return
    restoreMessages(detail.messages || [])
  } catch (err) {
    if (serial === historyRequestSerial) {
      rememberConversation(null)
      messages.value = []
      error.value = err?.message || '会话历史加载失败'
    }
  } finally {
    if (serial === historyRequestSerial) historyLoading.value = false
  }
}

async function restoreConversation() {
  const serial = ++historyRequestSerial
  historyLoading.value = true
  try {
    const items = await listWorkbookConversations(props.workbook.workbook_id)
    if (serial !== historyRequestSerial) return
    conversations.value = Array.isArray(items) ? items : []
    const saved = savedConversationId()
    const selected = conversations.value.find((item) => item.conv_id === saved)
      || conversations.value[0]
    rememberConversation(selected?.conv_id || null)
    await loadConversation(selected?.conv_id || null)
  } catch (err) {
    if (serial !== historyRequestSerial) return
    conversations.value = []
    rememberConversation(null)
    error.value = err?.message || '会话列表加载失败'
  } finally {
    if (serial === historyRequestSerial) historyLoading.value = false
  }
}

async function refreshConversationList() {
  try {
    const items = await listWorkbookConversations(props.workbook.workbook_id)
    conversations.value = Array.isArray(items) ? items : []
  } catch {
    // The current answer remains usable if refreshing the sidebar fails.
  }
}

async function switchConversation(id) {
  if (status.value === 'thinking' || !id) return
  error.value = ''
  rememberConversation(id)
  await loadConversation(id)
}

function startNewConversation() {
  cancelRequest()
  error.value = ''
  rememberConversation(null)
  messages.value = []
}

async function send() {
  const question = input.value.trim()
  if (!question || status.value === 'thinking') return
  input.value = ''
  error.value = ''
  status.value = 'thinking'

  const userMsg = { role: 'user', text: question }
  const asstMsg = { role: 'assistant', text: '', tables: [], reports: [] }
  messages.value.push(userMsg, asstMsg)

  const { promise, controller } = askWorkbookStream(
    props.workbook.workbook_id, question, 200, convId.value
  )
  activeController = controller

  try {
    const response = await validateSseResponse(await promise)
    if (activeController !== controller) return
    const consume = consumeStream(response, {
      onToken: (token) => {
        if (token === '__status__thinking') return
        asstMsg.text += token
      },
      onEvent: (event) => {
        if (event.conv_id) rememberConversation(event.conv_id)
        if (event.type === 'sql' && event.error) {
          asstMsg.text += `\n[SQL 错误] ${event.error}\n`
        } else if (event.type === 'table') {
          asstMsg.tables.push({
            columns: event.columns || [],
            rows: (event.rows || []).map((r, idx) => {
              const obj = {}
              ;(event.columns || []).forEach((c, ci) => { obj[c] = r[ci] })
              obj.__row__ = idx + 1
              return obj
            }),
            row_count: event.row_count || 0,
          })
        } else if (event.type === 'html' && !event.error) {
          asstMsg.reports.push({
            reportId: event.report_id,
            title: event.title || '分析报告',
          })
        }
      },
      onDone: (result) => {
        if (result.convId) rememberConversation(result.convId)
        if (result.fullAnswer && !asstMsg.text) asstMsg.text = result.fullAnswer
        if (result.reportUrl && result.reportTitle && !asstMsg.reports.length) {
          const rid = reportIdFromUrl(result.reportUrl)
          asstMsg.reports.push({ reportId: rid, title: result.reportTitle })
        }
        refreshConversationList()
        settleRequest()
        scrollBottom()
      },
      onError: (err) => {
        error.value = err?.message || '请求失败'
        settleRequest()
      },
    })
    activeConsumer = consume
  } catch (err) {
    if (activeController !== controller) return
    if (err?.name !== 'AbortError') error.value = err?.message || '请求失败'
    settleRequest()
  }
}

function settleRequest() {
  status.value = 'idle'
  activeConsumer = null
  activeController = null
}

function cancelRequest() {
  activeController?.abort()
  activeConsumer?.cancel()
  settleRequest()
}

watch(
  () => props.workbook.workbook_id,
  () => {
    cancelRequest()
    historyRequestSerial += 1
    conversations.value = []
    rememberConversation(null)
    messages.value = []
    status.value = 'idle'
    error.value = ''
    restoreConversation()
  }
)

onMounted(restoreConversation)
onBeforeUnmount(cancelRequest)

function scrollBottom() {
  nextTick(() => {
    if (msgArea.value) msgArea.value.scrollTop = msgArea.value.scrollHeight
  })
}
</script>

<style scoped>
.excel-qa {
  height: 100%;
  display: flex;
}
.qa-card {
  flex: 1;
  display: flex;
  flex-direction: column;
  border-radius: 8px;
}
.qa-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 0 12px;
  gap: 12px;
}
.wb-name {
  font-weight: 600;
  font-size: 14px;
}
.qa-session-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}
.conversation-select {
  width: 190px;
}
.msg-area {
  flex: 1;
  overflow-y: auto;
  padding: 8px 4px;
  min-height: 300px;
  max-height: 460px;
}
.welcome { color: #909399; text-align: center; padding: 48px 0; }
.hint { color: #b0b3b8; font-size: 12px; margin-top: 8px; }
.msg { margin: 10px 0; }
.msg-user { text-align: right; }
.msg-text {
  display: inline-block;
  max-width: 90%;
  padding: 8px 12px;
  border-radius: 8px;
  background: #f0f2f5;
  font-size: 13px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}
.msg-user .msg-text { background: #0f9c8f; color: #fff; }
.msg-assistant .msg-text { background: #f0f2f5; }
.table-block { margin: 8px 0; }
.table-caption { font-size: 12px; color: #909399; margin: 4px 0; }
.status-line { color: #909399; font-size: 13px; padding: 8px 0; }
.error-line { color: #f56c6c; font-size: 13px; padding: 8px 0; }
.input-bar { padding-top: 12px; }

@media (max-width: 760px) {
  .qa-header {
    align-items: flex-start;
    flex-direction: column;
  }

  .qa-session-bar,
  .conversation-select {
    width: 100%;
  }
}
</style>
