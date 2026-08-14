<template>
  <div class="excel-qa">
    <el-card shadow="never" class="qa-card">
      <div class="qa-header">
        <span class="wb-name">📊 {{ workbook.display_name || '工作簿' }}</span>
        <el-tag size="small" type="success">已建库 v{{ workbook.active_version }}</el-tag>
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
import { nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { askWorkbookStream } from '../api.js'
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
let activeConsumer = null
let activeController = null

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
        if (event.conv_id) convId.value = event.conv_id
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
        if (result.convId) convId.value = result.convId
        if (result.fullAnswer && !asstMsg.text) asstMsg.text = result.fullAnswer
        if (result.reportUrl && result.reportTitle && !asstMsg.reports.length) {
          const rid = result.reportUrl.split('/').pop().replace(/\.html$/, '')
          asstMsg.reports.push({ reportId: rid, title: result.reportTitle })
        }
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
    convId.value = null
    messages.value = []
    status.value = 'idle'
    error.value = ''
  }
)

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
}
.wb-name {
  font-weight: 600;
  font-size: 14px;
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
</style>
