<template>
  <div class="ai-layout">
    <!-- Sidebar -->
    <aside class="ai-sidebar">
      <div class="sidebar-header">
        <span>历史对话</span>
        <el-icon class="new-chat-icon" @click="handleNewChat"><Plus /></el-icon>
      </div>
      <!-- Loading -->
      <div v-if="convsLoading" style="text-align:center;padding:20px;color:#909399;">加载中...</div>
      <!-- Error -->
      <div v-else-if="convsError" class="sidebar-error">
        <p>{{ convsError }}</p>
        <el-button size="small" @click="loadConversations">重试</el-button>
      </div>
      <!-- List -->
      <template v-else>
        <div v-if="!messageGroups.length" class="sidebar-empty">
          <p>暂无对话记录</p>
          <p style="font-size:12px;color:#909399">点击 + 开始新对话</p>
        </div>
        <div v-for="group in messageGroups" :key="group.date" class="chat-group">
          <div class="group-date">{{ group.date }}</div>
          <div
            v-for="chat in group.items"
            :key="chat.conv_id"
            :class="['chat-item', { active: chat.conv_id === activeConvId }]"
            @click="switchConversation(chat.conv_id)"
          >
            <el-icon><ChatDotRound /></el-icon>
            <span class="chat-title">{{ chat.title || '新对话' }}</span>
            <span class="chat-count">{{ chat.message_count || 0 }}</span>
            <el-icon class="chat-delete" @click.stop="handleDeleteConv(chat.conv_id)"><Close /></el-icon>
          </div>
        </div>
      </template>
    </aside>

    <!-- Main -->
    <main class="ai-main">
      <!-- Chat Area -->
      <div class="ai-chat-area" ref="chatAreaRef">
        <!-- Greeting (no active conversation and no messages) -->
        <div v-if="!activeConvId && messages.length === 0" class="ai-greeting">
          <p class="greeting-text">{{ greeting }}</p>
          <div class="quick-actions">
            <span
              v-for="action in quickActions"
              :key="action"
              class="quick-chip"
              @click="inputText = action"
            >{{ action }}</span>
          </div>
        </div>

        <!-- Loading conversation messages -->
        <div v-if="msgsLoading" style="text-align:center;padding:40px;color:#909399;">加载对话...</div>

        <!-- Messages -->
        <div v-for="(msg, i) in messages" :key="i" :class="['msg-row', msg.role]">
          <div :class="['msg-bubble', msg.role]">
            <div v-if="msg.role === 'user'">{{ msg.content }}</div>
            <div v-else>
              <!-- Status markers (non-display) -->
              <div v-if="msg.content.startsWith('⏳') && msg.content.length < 20" class="status-msg">
                🔍 检索中...
              </div>
              <div v-else-if="msg.content.startsWith('thinking...') && msg.content.length < 20" class="status-msg">
                💭 思考中...
              </div>
              <!-- Content -->
              <div v-else class="assistant-content" v-html="renderMarkdown(msg.content)"></div>
              <!-- Streaming cursor -->
              <span v-if="i === messages.length - 1 && isStreaming" class="stream-cursor">▍</span>
              <!-- Sources (done, has data) -->
              <div v-if="msg.role === 'assistant' && msg.sources && msg.sources.length" class="msg-sources">
                <div class="sources-title">📚 引用来源</div>
                <div
                  v-for="(src, si) in msg.sources"
                  :key="si"
                  class="source-item"
                  @click="openChunks(src)"
                >
                  <span class="source-file">{{ src.file_path?.split('/').pop() || src.file_path }}</span>
                  <span v-if="src.doc_number" class="source-doc">({{ src.doc_number }})</span>
                  <span class="source-chunks"><el-icon><Document /></el-icon> {{ src.chunks?.length || 0 }} 个片段</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Input Panel -->
      <div class="ai-input-panel">
        <!-- Params row -->
        <div class="params-row">
          <el-select v-model="domainFilter" placeholder="全部域" clearable size="small" style="width:120px">
            <el-option label="全部域" value="" />
            <el-option label="变电" value="变电" />
            <el-option label="配电" value="配电" />
            <el-option label="送电输电" value="送电输电" />
            <el-option label="综合" value="综合" />
          </el-select>
          <span class="param-label">Top-K</span>
          <el-slider v-model="topK" :min="5" :max="30" :step="5" size="small" style="width:100px" show-stops />
          <span class="param-val">{{ topK }}</span>
        </div>

        <el-input
          v-model="inputText"
          type="textarea"
          :rows="3"
          placeholder="请输入您的问题..."
          resize="none"
          class="input-textarea"
          :disabled="isStreaming"
          @keydown.enter.exact.prevent="handleSend"
        />
        <div class="input-toolbar">
          <div class="toolbar-left">
            <!-- Mode dropdown (shell) -->
            <el-dropdown trigger="click" @command="handleModeChange">
              <span class="mode-btn mode-primary">
                {{ modeLabel }} <el-icon><ArrowDown /></el-icon>
              </span>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item command="quick">快速对话</el-dropdown-item>
                  <el-dropdown-item command="deep">深度思考</el-dropdown-item>
                  <el-dropdown-item command="web">联网搜索</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>

            <el-button
              :type="knowledgeShared ? 'primary' : 'default'"
              plain
              size="small"
              @click="knowledgeShared = !knowledgeShared"
            >
              {{ knowledgeShared ? '✓' : '' }} 共享知识库
            </el-button>

            <el-button plain size="small" class="sys-btn" disabled>金曲系统</el-button>
          </div>

          <div class="toolbar-right">
            <el-button
              v-if="isStreaming"
              type="danger"
              size="small"
              @click="handleStop"
            >停止</el-button>
            <el-button
              v-else
              type="primary"
              :icon="Promotion"
              circle
              :disabled="!inputText.trim()"
              class="send-btn"
              @click="handleSend"
            />
          </div>
        </div>
      </div>
    </main>

    <!-- Chunks Modal -->
    <el-dialog
      v-model="chunksModal.open"
      :title="'📄 ' + (chunksModal.filePath.split('/').pop() || chunksModal.filePath)"
      width="700px"
      top="5vh"
      destroy-on-close
    >
      <div v-if="!chunksModal.chunks.length" style="text-align:center;padding:40px;color:#909399;">无相关片段</div>
      <div v-else class="chunk-list">
        <div v-for="c in chunksModal.chunks" :key="c.chunk_id" class="chunk-card">
          <div class="chunk-meta">
            <el-tag size="small" v-if="c.page_num">第 {{ c.page_num }} 页</el-tag>
            <el-tag size="small" type="success" v-if="c.score">{{ (c.score * 100).toFixed(0) }}%</el-tag>
          </div>
          <div class="chunk-text">{{ c.text }}</div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick, watch } from 'vue'
import { Promotion } from '@element-plus/icons-vue'
import * as API from './api.js'
import { consumeStream } from './sse.js'

// ---- Greeting ----
const greeting = computed(() => {
  const h = new Date().getHours()
  if (h < 5) return '夜深了，注意休息 🌙'
  if (h < 12) return '上午好，今天有什么可以帮您？☀️'
  if (h < 18) return '下午好，随时为您服务 🌤'
  return '晚上好，还有什么需要解答的？🌙'
})

// ---- Conversations sidebar ----
const conversations = ref([])
const convsLoading = ref(false)
const convsError = ref('')
const activeConvId = ref(null)

async function loadConversations() {
  convsLoading.value = true
  convsError.value = ''
  const res = await API.listConversations()
  if (typeof res === 'string') { convsError.value = res; convsLoading.value = false; return }
  conversations.value = Array.isArray(res) ? res : []
  convsLoading.value = false
}

const messageGroups = computed(() => {
  const groups = {}
  const list = [...conversations.value].sort((a, b) => (b.updated_at || '').localeCompare(a.updated_at || ''))
  list.forEach(c => {
    const date = (c.updated_at || '').slice(0, 10) || '未知日期'
    if (!groups[date]) groups[date] = []
    groups[date].push(c)
  })
  return Object.entries(groups).map(([date, items]) => ({ date, items }))
})

async function handleNewChat() {
  // ponytail: don't block creating if > 0 conversations exist
  const res = await API.createConversation()
  if (typeof res === 'string') { console.error(res); return }
  conversations.value.unshift(res)
  activeConvId.value = res.conv_id
  messages.value = []
}

async function switchConversation(convId) {
  if (convId === activeConvId.value) return
  activeConvId.value = convId
  msgsLoading.value = true
  const res = await API.getConversation(convId)
  msgsLoading.value = false
  if (typeof res === 'string') { console.error(res); return }
  messages.value = (res.messages || []).map(m => ({
    role: m.role,
    content: m.content,
    citations: m.citations,
    sources: m.sources,
  }))
  await nextTick()
  scrollToBottom()
}

async function handleDeleteConv(convId) {
  const res = await API.deleteConversation(convId)
  if (typeof res === 'string') { console.error(res); return }
  conversations.value = conversations.value.filter(c => c.conv_id !== convId)
  if (activeConvId.value === convId) {
    activeConvId.value = null
    messages.value = []
  }
}

// ---- Messages + Streaming ----
const messages = ref([])
const msgsLoading = ref(false)
const inputText = ref('')
const isStreaming = ref(false)
const stopFn = ref(null)  // { cancelReader, abortFetch }

const topK = ref(15)
const domainFilter = ref('')
// ponytail: knowledgeShared = true means no domain filter (search all)
const knowledgeShared = ref(true)

const mode = ref('quick')
const modeLabel = computed(() =>
  ({ quick: '快速对话', deep: '深度思考', web: '联网搜索' })[mode.value]
)
// ponytail: these are shell UI, backend doesn't support them yet
const quickActions = ['投标文件生成', '初设报告生成']

// Chunks modal
const chunksModal = ref({ open: false, filePath: '', chunks: [] })
function openChunks(source) {
  chunksModal.value = { open: true, filePath: source.file_path || '', chunks: source.chunks || [] }
}
function closeChunks() { chunksModal.value.open = false }

const chatAreaRef = ref(null)

function scrollToBottom() {
  nextTick(() => {
    const el = chatAreaRef.value
    if (el) el.scrollTop = el.scrollHeight
  })
}

async function handleSend() {
  const text = inputText.value.trim()
  if (!text || isStreaming.value) return

  // Auto-create conversation if none active
  if (!activeConvId.value) {
    const res = await API.createConversation(text.slice(0, 30))
    if (typeof res === 'string') { console.error(res); return }
    conversations.value.unshift(res)
    activeConvId.value = res.conv_id
  }

  // Push user message
  messages.value.push({ role: 'user', content: text })
  inputText.value = ''
  // Push empty assistant placeholder
  messages.value.push({ role: 'assistant', content: '' })
  const assistantIdx = messages.value.length - 1

  isStreaming.value = true
  await nextTick()
  scrollToBottom()

  // Determine domain filter: knowledgeShared=true means all, else use selected
  const domain = knowledgeShared.value ? null : (domainFilter.value || null)

  // Start SSE stream
  const { promise, controller } = API.askStream(text, topK.value, domain, activeConvId.value)
  // Save stop handles
  const abortFetch = () => controller.abort()
  stopFn.value = { abortFetch, cancelReader: null }

  try {
    const resp = await promise
    stopFn.value.abortFetch = null  // fetch already resolved
    if (!resp.ok) {
      messages.value[assistantIdx].content = `⚠ API 错误 (HTTP ${resp.status})`
      isStreaming.value = false
      stopFn.value = null
      return
    }
    const reader = consumeStream(resp, {
      onToken(token) {
        if (token === '__status__searching') {
          messages.value[assistantIdx].content = '⏳ 检索中...'
          return
        }
        if (token === '__status__thinking') {
          messages.value[assistantIdx].content = '💭 思考中...'
          return
        }
        const cur = messages.value[assistantIdx].content
        if (cur.startsWith('⏳') || cur.startsWith('💭')) {
          messages.value[assistantIdx].content = token
        } else {
          messages.value[assistantIdx].content += token
        }
        scrollToBottom()
      },
      onDone(result) {
        messages.value[assistantIdx].content = result.fullAnswer || messages.value[assistantIdx].content
        messages.value[assistantIdx].sources = result.sources || []
        messages.value[assistantIdx].citations = result.citations || []
        isStreaming.value = false
        stopFn.value = null
        scrollToBottom()
        loadConversations()
      },
      onError(err) {
        messages.value[assistantIdx].content = `[错误] ${err.message || err}`
        isStreaming.value = false
        stopFn.value = null
      },
    })
    stopFn.value = { abortFetch: null, cancelReader: reader.cancel }
  } catch (e) {
    messages.value[assistantIdx].content = `[错误] ${e.message || e}`
    isStreaming.value = false
    stopFn.value = null
  }
}

function handleStop() {
  const s = stopFn.value
  if (s) {
    if (s.abortFetch) s.abortFetch()
    if (s.cancelReader) s.cancelReader()
  }
  isStreaming.value = false
  stopFn.value = null
}

function handleModeChange(cmd) { mode.value = cmd }
// ponytail: mode is UI-only shell, backend only does one mode (RAG)

// ---- Markdown render (ponytail: minimal, no lib) ----
function renderMarkdown(text) {
  if (!text) return ''
  let html = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  // Code blocks
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>')
  // Bold
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  // Headers
  html = html.replace(/^### (.+)$/gm, '<h4>$1</h4>')
  html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>')
  // Lists
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>')
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>')
  // Numbered lists
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
  // Paragraphs (double newline)
  html = html.replace(/\n\n/g, '</p><p>')
  html = '<p>' + html + '</p>'
  // Clean empty paragraphs
  html = html.replace(/<p>\s*<\/p>/g, '')
  return html
}

// ---- Init ----
onMounted(() => {
  loadConversations()
  // ponytail: watch knowledgeShared to sync with domainFilter
  watch(knowledgeShared, (val) => {
    if (!val && !domainFilter.value) domainFilter.value = ''
  })
})
</script>

<style scoped>
.ai-layout { display: flex; height: 100%; }

/* ---- Sidebar ---- */
.ai-sidebar {
  width: 260px; min-width: 260px;
  background: #fff;
  border-right: 1px solid #ebeef0;
  overflow-y: auto;
  display: flex; flex-direction: column;
}
.sidebar-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 20px 12px;
  font-size: 15px; font-weight: 600; color: #303133;
  border-bottom: 1px solid #f0f0f0;
}
.new-chat-icon { cursor: pointer; color: #0f9c8f; font-size: 18px; }
.sidebar-error { padding: 16px; color: #f56c6c; font-size: 13px; text-align: center; }
.sidebar-empty { padding: 30px 16px; text-align: center; color: #909399; font-size: 13px; }

.chat-group { padding: 0 12px; }
.group-date { padding: 12px 8px 6px; font-size: 12px; color: #909399; }
.chat-item {
  display: flex; align-items: center; gap: 8px;
  padding: 10px 8px; border-radius: 6px;
  cursor: pointer; font-size: 13px; color: #606266;
  transition: background 0.15s;
  position: relative;
}
.chat-item:hover { background: #f2f8f7; }
.chat-item.active { background: #e6f7f5; color: #0f9c8f; font-weight: 500; }
.chat-title { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.chat-count { font-size: 11px; color: #909399; min-width: 16px; text-align: center; }
.chat-delete {
  display: none; font-size: 14px; color: #909399;
  position: absolute; right: 8px;
}
.chat-item:hover .chat-delete { display: inline-flex; }
.chat-delete:hover { color: #f56c6c; }

/* ---- Main ---- */
.ai-main { flex: 1; display: flex; flex-direction: column; background: #fff; }
.ai-chat-area { flex: 1; padding: 24px 40px; overflow-y: auto; }

/* Greeting */
.ai-greeting {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; height: 100%; gap: 24px;
}
.greeting-text { font-size: 22px; color: #1f2329; }
.quick-actions { display: flex; gap: 12px; }
.quick-chip {
  padding: 8px 20px; border: 1px solid #0f9c8f;
  border-radius: 20px; color: #0f9c8f;
  cursor: pointer; font-size: 14px; transition: all 0.2s;
}
.quick-chip:hover { background: #e6f7f5; }

/* Messages */
.msg-row { display: flex; margin-bottom: 16px; }
.msg-row.user { justify-content: flex-end; }
.msg-row.assistant { justify-content: flex-start; }
.msg-bubble {
  max-width: 70%; padding: 12px 16px;
  border-radius: 10px; font-size: 14px; line-height: 1.6; word-break: break-word;
}
.msg-bubble.user { background: #e3f6f3; color: #303133; }
.msg-bubble.assistant { background: #f2f3f5; color: #303133; }

.status-msg { color: #909399; font-style: italic; }
.stream-cursor {
  animation: blink 0.8s infinite;
  color: #0f9c8f; font-weight: bold;
}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

/* Markdown styles */
.assistant-content :deep(p) { margin: 0 0 8px; }
.assistant-content :deep(p:last-child) { margin: 0; }
.assistant-content :deep(ul), .assistant-content :deep(ol) { margin: 4px 0; padding-left: 20px; }
.assistant-content :deep(li) { margin: 2px 0; }
.assistant-content :deep(pre) { background: #f0f0f0; padding: 10px; border-radius: 6px; overflow-x: auto; margin: 8px 0; font-size: 13px; }
.assistant-content :deep(code) { background: #f0f0f0; padding: 2px 5px; border-radius: 3px; font-size: 13px; }
.assistant-content :deep(pre code) { background: none; padding: 0; }
.assistant-content :deep(h3) { margin: 12px 0 6px; font-size: 15px; }
.assistant-content :deep(h4) { margin: 8px 0 4px; font-size: 14px; }
.assistant-content :deep(strong) { font-weight: 600; }

/* Sources */
.msg-sources {
  margin-top: 10px; padding-top: 10px;
  border-top: 1px solid #e8e8e8;
}
.sources-title { font-size: 12px; color: #909399; margin-bottom: 6px; font-weight: 600; }
.source-item {
  display: flex; align-items: center; gap: 8px;
  padding: 6px 8px; font-size: 12px; color: #606266;
  border-radius: 6px; cursor: pointer; transition: background 0.15s;
}
.source-item:hover { background: #e6f7f5; }
.source-file { color: #0f9c8f; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.source-doc { color: #909399; font-size: 11px; }
.source-chunks { color: #909399; font-size: 11px; display: flex; align-items: center; gap: 3px; }

/* Chunks in modal */
.chunk-list { display: flex; flex-direction: column; gap: 12px; }
.chunk-card {
  background: #fafafa; border: 1px solid #ebeef0;
  border-radius: 8px; padding: 12px 14px;
}
.chunk-meta { display: flex; gap: 8px; margin-bottom: 8px; }
.chunk-text { font-size: 13px; line-height: 1.7; color: #303133; white-space: pre-wrap; }

/* Input Panel */
.ai-input-panel {
  border-top: 1px solid #ebeef0;
  padding: 12px 40px 20px; max-width: 1180px; width: 100%; margin: 0 auto;
}
.params-row {
  display: flex; align-items: center; gap: 8px;
  margin-bottom: 8px; font-size: 12px; color: #909399;
}
.param-label { min-width: 36px; }
.param-val { min-width: 20px; text-align: right; }
.input-textarea :deep(.el-textarea__inner) {
  border-radius: 4px 12px 12px 12px;
  border-color: #dcdfe6;
}
.input-toolbar {
  display: flex; align-items: center; justify-content: space-between;
  margin-top: 10px;
}
.toolbar-left { display: flex; align-items: center; gap: 8px; }
.toolbar-right { display: flex; align-items: center; }

.mode-btn {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 4px 14px; border-radius: 16px;
  font-size: 13px; cursor: pointer;
  border: 1px solid #0f9c8f; color: #0f9c8f;
  background: rgba(15,156,143,.06);
}
.sys-btn { border-color: #dcdfe6 !important; color: #909399 !important; }
.send-btn {
  width: 38px; height: 38px;
  background: #0f9c8f !important; border-color: #0f9c8f !important;
}
.send-btn:hover { background: #0d8a7e !important; }
</style>
