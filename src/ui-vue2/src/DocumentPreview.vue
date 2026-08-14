<template>
  <div class="dp-layout">
    <!-- Top toolbar -->
    <header class="dp-toolbar">
      <el-button class="dp-toolbar__left-btn" :disabled="saving" @click="isEditing ? cancelEdit() : $emit('close')">
        {{ isEditing ? '取消编辑' : '关闭' }}
      </el-button>
      <div class="dp-toolbar__center">
        <span class="dp-toolbar__title">{{ fileData.file_name }}</span>
        <span v-if="fileData.doc_number" class="dp-toolbar__docnum">{{ fileData.doc_number }}</span>
        <span v-if="isEditing" class="dp-toolbar__editing">编辑中 · {{ editRecords.length }} 处修改</span>
      </div>
      <div class="dp-toolbar__actions">
        <el-button v-if="!isEditing" plain @click="runOcrCompare(true)" :loading="ocrLoading">OCR 对比</el-button>
        <el-button type="primary" plain class="dp-toolbar__right-btn" :loading="saving || editLoading" @click="isEditing ? saveEdit() : startEdit()">
          {{ isEditing ? '保存并同步' : '编辑' }}
        </el-button>
      </div>
    </header>

    <div class="dp-body">
      <!-- Left: page navigation -->
      <aside class="dp-nav">
        <div class="dp-nav__title">目录导航</div>
        <div v-if="loading" class="dp-nav__loading">加载中...</div>
        <template v-else>
          <div
            v-for="(item, idx) in catalog"
            :key="item.id"
            :class="['dp-nav__item', { 'is-active': activeCatalog === item.id, 'is-root': Number(item.level) === 0 }]"
            :style="{ paddingLeft: `${Math.min(6, Math.max(0, Number(item.level) || 0)) * 14 + 8}px` }"
            @click="scrollTo(item.id)"
          >
            <span class="dp-nav__item-text">{{ item.title }}</span>
            <span v-if="item.page" class="dp-nav__item-page">P{{ item.page }}</span>
          </div>
          <div v-if="!catalog.length" class="dp-nav__empty">暂无目录</div>
        </template>
      </aside>

      <!-- Center: document content (按页渲染, 目录锚点跳转) -->
      <main class="dp-content" ref="contentRef">
        <div v-if="loading" class="dp-content__paper" style="color:#909399">加载中...</div>
        <div v-else-if="error" class="dp-content__paper" style="color:#f56c6c">{{ error }}</div>
        <template v-else>
          <section v-if="!isEditing && showOcrCompare && ocrCompare" class="dp-ocr-compare">
            <div class="dp-ocr-compare__head">
              <strong>OCR 结果对比</strong>
              <el-button link @click="showOcrCompare = false">关闭对比</el-button>
            </div>
            <div class="dp-ocr-compare__grid">
              <article class="dp-ocr-card">
                <h3>8080 裸 PaddleOCR-VL</h3>
                <el-alert v-if="ocrCompare.bare_8080 && !ocrCompare.bare_8080.ok" :title="ocrCompare.bare_8080.error" type="error" :closable="false" />
                <div v-else-if="ocrCompare.bare_8080?.pages?.length" class="dp-ocr-pages">
                  <section v-for="page in ocrCompare.bare_8080.pages" :key="`bare-page-${page.page_num}`" class="dp-ocr-page">
                    <h4>第 {{ page.page_num }} 页</h4>
                    <pre>{{ page.text || '无识别结果' }}</pre>
                  </section>
                </div>
                <pre v-else>{{ ocrCompare.bare_8080?.text || '无识别结果' }}</pre>
              </article>
              <article class="dp-ocr-card">
                <h3>8001 PaddleOCR Pipeline</h3>
                <el-alert v-if="ocrCompare.pipeline_8001 && !ocrCompare.pipeline_8001.ok" :title="ocrCompare.pipeline_8001.error" type="error" :closable="false" />
                <div v-else-if="ocrCompare.pipeline_8001?.pages?.length" class="dp-ocr-pages">
                  <section v-for="page in ocrCompare.pipeline_8001.pages" :key="`pipeline-page-${page.page_num}`" class="dp-ocr-page">
                    <h4>第 {{ page.page_num }} 页</h4>
                    <iframe v-if="page.layout_html" class="dp-ocr-frame" scrolling="no" sandbox="allow-scripts" :data-page-num="page.page_num" :srcdoc="page.layout_html"></iframe>
                    <pre v-else>{{ page.text || '无识别结果' }}</pre>
                  </section>
                </div>
                <iframe v-else-if="ocrCompare.pipeline_8001?.layout_html" class="dp-ocr-frame" scrolling="no" sandbox="allow-scripts" :srcdoc="ocrCompare.pipeline_8001.layout_html"></iframe>
                <iframe v-else-if="ocrCompare.pipeline_8001?.text" class="dp-ocr-frame" sandbox :srcdoc="ocrCompare.pipeline_8001.text"></iframe>
                <div v-else class="dp-ocr-empty">无识别结果</div>
              </article>
            </div>
          </section>
          <section v-if="!isEditing && ocrCompare?.pipeline_8001?.pages?.length" class="dp-primary-layout">
            <div
              v-for="page in ocrCompare.pipeline_8001.pages"
              :key="`compare-layout-page-${page.page_num}`"
              :id="`page-${page.page_num}`"
              class="dp-primary-layout__page"
            >
              <div class="dp-primary-layout__title">第 {{ page.page_num }} 页</div>
              <iframe v-if="page.layout_html" class="dp-primary-layout__frame" scrolling="no" sandbox="allow-scripts" :data-page-num="page.page_num" :srcdoc="page.layout_html"></iframe>
              <div v-else class="dp-ocr-empty">本页无版面结果</div>
            </div>
          </section>
          <section v-else-if="displayLayoutPages.length" :class="['dp-primary-layout', { 'is-editing': isEditing }]">
            <div
              v-for="page in displayLayoutPages"
              :key="`cached-layout-page-${page.page_num}`"
              :id="`page-${page.page_num}`"
              :class="['dp-primary-layout__page', { 'is-editing': isEditing }]"
            >
              <div class="dp-primary-layout__title">
                <span>第 {{ page.page_num }} 页</span>
                <span v-if="isEditing" class="dp-primary-layout__hint">点击文字原位修改 · 悬停图片可删除</span>
              </div>
              <iframe v-if="page.layout_html" class="dp-primary-layout__frame" scrolling="no" sandbox="allow-scripts" :data-page-num="page.page_num" :srcdoc="page.layout_html"></iframe>
              <div v-else class="dp-ocr-empty">本页无版面结果</div>
            </div>
          </section>
          <section v-else-if="!isEditing && cachedLayoutHtml" class="dp-primary-layout">
            <iframe class="dp-primary-layout__frame" scrolling="no" sandbox="allow-scripts" :srcdoc="cachedLayoutHtml"></iframe>
          </section>
          <div
            v-else-if="!isEditing"
            v-for="block in pageBlocks"
            :key="block.page"
            :id="`page-${block.page}`"
            class="dp-content__paper dp-page"
          >
            <div class="dp-page__title">## 第 {{ block.page }} 页</div>
            <div class="dp-page__body" v-html="renderPreviewHtml(block.text)"></div>
          </div>
          <div v-else class="dp-content__paper dp-edit-unavailable">当前文件没有可编辑的版面缓存，请先重新解析该文件。</div>
        </template>
      </main>

      <!-- Right: edit history -->
      <aside class="dp-side">
        <div class="dp-side__header" @click="recordsCollapsed = !recordsCollapsed">
          <span>当前共计 <span class="dp-side__count">{{ editRecords.length }}</span> 处待保存修改</span>
          <el-icon class="dp-side__toggle" :class="{ 'is-collapsed': recordsCollapsed }">
            <ArrowUp />
          </el-icon>
        </div>

        <div v-show="!recordsCollapsed" class="dp-side__list">
          <div v-if="editRecords.length === 0" class="dp-side__empty">暂无编辑记录</div>

          <div v-for="(record, idx) in editRecords" :key="record.id" class="dp-record">
            <div class="dp-record__header">
              <span class="dp-record__title">编辑记录{{ idx + 1 }}</span>
              <span v-if="isEditing && !saving" class="dp-record__restore" @click="restoreRecord(record)">恢复</span>
            </div>
            <div class="dp-record__label">修改前：</div>
            <div class="dp-record__before">{{ record.before }}</div>
            <div class="dp-record__label">修改后：</div>
            <div class="dp-record__after">{{ record.after }}</div>
          </div>
        </div>
      </aside>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onBeforeUnmount, nextTick } from 'vue'
import { ArrowUp } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import * as API from './api.js'

const props = defineProps({
  fileHash: { type: String, required: true },
  fileName: { type: String, default: '' },
})

defineEmits(['close'])

// ── Fetch file content ──
const loading = ref(true)
const error = ref('')
const ocrLoading = ref(false)
const ocrCompare = ref(null)
const showOcrCompare = ref(false)
const cachedLayoutHtml = ref('')
const cachedLayoutPages = ref([])
const editableLayoutPages = ref([])
const layoutRevision = ref('')
const fileData = reactive({
  file_name: props.fileName,
  doc_number: '',
  domain: '',
  category: '',
  full_text: '',
  chunks: [],
  total_chunks: 0,
  total_pages: 0,
  layout_html: '',
})

onMounted(async () => {
  window.addEventListener('message', onLayoutFrameMessage)
  await loadFileContent()
})

async function loadFileContent() {
  loading.value = true
  error.value = ''
  try {
    const data = await API.getFileContent(props.fileHash)
    Object.assign(fileData, data)
    layoutRevision.value = data.layout_revision || ''
    // The backend now renders cached 8001 blocks with the exact same renderer
    // used by the live OCR comparison. Keep the old client renderer only as a
    // compatibility fallback for older API servers/cache records.
    cachedLayoutPages.value = Array.isArray(data.layout_pages)
      ? data.layout_pages.filter(page => page && Number(page.page_num) > 0)
      : []
    if (!cachedLayoutPages.value.length && data.layout_blocks) {
      if (Array.isArray(data.layout_blocks)) {
        const pageBlocks = data.layout_blocks.filter(block => block && block.page !== undefined && block.page !== null)
        if (pageBlocks.length) {
          const grouped = new Map()
          for (const block of pageBlocks) {
            const page = Number(block.page) + 1
            if (!Number.isFinite(page) || page < 1) continue
            if (!grouped.has(page)) grouped.set(page, [])
            grouped.get(page).push(block)
          }
          cachedLayoutPages.value = [...grouped.entries()]
            .sort(([a], [b]) => a - b)
            .map(([page_num, blocks]) => ({ page_num, layout_html: renderCachedLayout(blocks) }))
        } else {
          const html = renderCachedLayout(data.layout_blocks)
          if (html) cachedLayoutPages.value = [{ page_num: 1, layout_html: html }]
        }
      } else if (typeof data.layout_blocks === 'object') {
        cachedLayoutPages.value = Object.entries(data.layout_blocks)
          .map(([page, blocks]) => ({
            page_num: Number(page) + 1,
            layout_html: renderCachedLayout(blocks || []),
          }))
          .filter(page => page.page_num > 0)
      }
    }
    cachedLayoutHtml.value = data.layout_html || (
      cachedLayoutPages.value.length === 1
        ? cachedLayoutPages.value[0].layout_html
        : ''
    )
    // 按页聚合 chunk → 渲染为分页块
    pageBlocks.splice(0, pageBlocks.length)
    const pageMap = new Map()
    for (const c of (data.chunks || [])) {
      const p = c.page_num || 0
      if (!pageMap.has(p)) pageMap.set(p, { page: p, text: [] })
      pageMap.get(p).text.push(c.text)
    }
    for (const { page, text } of pageMap.values()) {
      pageBlocks.push({ page, text: mergeChunkOverlap(text) })
    }
    rebuildCatalog(data.outline)
    loading.value = false
    await nextTick()
    renderPageMath()
  } catch (e) {
    error.value = '加载失败: ' + (e.message || e)
    loading.value = false
  }
}

onBeforeUnmount(() => {
  window.removeEventListener('message', onLayoutFrameMessage)
})

function onLayoutFrameMessage(event) {
  const data = event?.data
  if (!data) return
  const frames = contentRef.value?.querySelectorAll('iframe') || []
  const frame = [...frames].find(item => item.contentWindow === event.source)
  if (!frame) return
  if (data.type === 'rongneng-layout-edit') {
    if (!isEditing.value || saving.value) return
    const pageNum = Number(data.page_num)
    const blockIndex = Number(data.block_index)
    if (!Number.isInteger(pageNum) || pageNum < 1 || !Number.isInteger(blockIndex) || blockIndex < 0) return
    const key = `${pageNum}:${blockIndex}`
    const before = String(data.before || '')
    const op = data.op === 'delete' ? 'delete' : 'update'
    const content = String(data.content || '')
    if (op === 'update' && content.trim() === before.trim()) {
      pendingEdits.delete(key)
      return
    }
    pendingEdits.set(key, {
      id: key,
      page_num: pageNum,
      block_index: blockIndex,
      op,
      content,
      content_format: data.content_format === 'html' ? 'html' : 'text',
      before: before || (op === 'delete' ? '[图片]' : ''),
      after: op === 'delete' ? '[已删除图片及图内识别内容]' : content,
    })
    return
  }
  if (data.type === 'rongneng-layout-anchor' && Number.isFinite(Number(data.offset))) {
    const content = contentRef.value
    if (!content) return
    const contentRect = content.getBoundingClientRect()
    const frameRect = frame.getBoundingClientRect()
    const top = content.scrollTop + frameRect.top - contentRect.top + Number(data.offset) - 18
    content.scrollTo({ top: Math.max(0, top), behavior: 'smooth' })
    return
  }
  if (data.type !== 'rongneng-layout-size' || !Number.isFinite(Number(data.height))) return
  frame.style.height = `${Math.max(160, Math.ceil(Number(data.height)))}px`
}

async function runOcrCompare(reveal = true) {
  ocrLoading.value = true
  if (reveal) showOcrCompare.value = true
  try {
    const data = await API.compareFileOcr(props.fileHash)
    ocrCompare.value = data
    if (data.pipeline_8001?.outline?.length) rebuildCatalog(data.pipeline_8001.outline)
  } catch (e) {
    ElMessage.error(`OCR 对比失败：${e.message || e}`)
  } finally {
    ocrLoading.value = false
  }
}

// ── Catalog ──
const pageBlocks = reactive([])     // 分页渲染块
const catalog = reactive([])
const activeCatalog = ref('')
const contentRef = ref(null)

function rebuildCatalog(outline = []) {
  const headings = Array.isArray(outline)
    ? outline
      .filter(item => item && Number(item.page) > 0 && item.title)
      .map((item, index) => ({
        id: item.id || `outline-${index}`,
        anchor: item.anchor || '',
        title: String(item.title),
        level: Math.max(0, Number(item.level) || 0),
        page: Number(item.page),
      }))
    : []
  if (headings.length) {
    catalog.splice(0, catalog.length, ...headings)
  } else {
    const pageNumbers = new Set(
      pageBlocks.map(b => Number(b.page)).filter(page => page > 0),
    )
    for (const page of cachedLayoutPages.value) pageNumbers.add(Number(page.page_num))
    catalog.splice(0, catalog.length,
      ...[...pageNumbers].sort((a, b) => a - b)
        .map(page => ({ id: `page-${page}`, anchor: '', title: `第 ${page} 页`, level: 0, page })))
  }
  if (catalog.length > 0) activeCatalog.value = catalog[0].id
}

function scrollTo(id) {
  activeCatalog.value = id
  const item = catalog.find(entry => entry.id === id)
  nextTick(() => {
    if (item?.anchor) {
      const frames = contentRef.value?.querySelectorAll('.dp-primary-layout__frame[data-page-num]') || []
      const frame = [...frames].find(entry => Number(entry.dataset.pageNum) === Number(item.page))
      if (frame) {
        frame.scrollIntoView({ behavior: 'smooth', block: 'start' })
        const send = () => frame.contentWindow?.postMessage({
          type: 'rongneng-layout-scroll',
          anchor: item.anchor,
        }, '*')
        send()
        frame.addEventListener('load', send, { once: true })
        window.setTimeout(send, 80)
        return
      }
    }
    const el = contentRef.value?.querySelector(`#${id}`)
    el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  })
}

function renderPageMath() {
  if (!contentRef.value) return
  const mathJax = window.MathJax
  if (mathJax?.typesetPromise) mathJax.typesetPromise([contentRef.value]).catch(() => {})
}

function renderCachedLayout(blocks) {
  const valid = (blocks || []).filter(b => {
    const box = b?.bbox || b?.block_bbox
    return Array.isArray(box) && box.length >= 4 && b.block_content
  })
  if (!valid.length) return ''
  const getBox = b => b.bbox || b.block_bbox
  const minX = Math.min(...valid.map(b => Number(getBox(b)[0]) || 0))
  const maxX = Math.max(1, ...valid.map(b => Number(getBox(b)[2]) || 0))
  const pageWidth = maxX + Math.max(0, minX)
  const maxY = Math.max(1, ...valid.map(b => Number(getBox(b)[3]) || 0))
  const esc = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))
  const items = valid.map(b => {
    const [x1, y1, x2, y2] = getBox(b).map(Number)
    return `<div style="position:absolute;left:${x1}px;top:${y1}px;width:${Math.max(1, x2-x1)}px;height:${Math.max(1, y2-y1)}px;white-space:pre-wrap;overflow:hidden;box-sizing:border-box"><div style="width:100%;height:100%;overflow:hidden">${esc(b.block_content)}</div></div>`
  }).join('')
  const pageFontSize = Math.min(48, Math.max(14, pageWidth * 0.015)).toFixed(1)
  return `\x3Cscript>window.MathJax={tex:{inlineMath:[["$","$"],["\\\\(","\\\\)"]],displayMath:[["$$","$$"],["\\\\[","\\\\]"]]}};(function(){function fit(){var page=document.querySelector(".page");if(!page)return;var scale=Math.min(1,Math.max(320,window.innerWidth-24)/page.offsetWidth);page.style.transformOrigin="top left";page.style.transform="scale("+scale+")";var h=page.offsetHeight*scale;page.style.marginLeft=Math.max(8,(window.innerWidth-page.offsetWidth*scale)/2)+"px";page.style.marginRight=Math.max(8,(window.innerWidth-page.offsetWidth*scale)/2)+"px";document.documentElement.style.overflow="hidden";document.body.style.overflow="hidden";document.body.style.height=(h+16)+"px";if(window.parent&&window.parent!==window)window.parent.postMessage({type:"rongneng-layout-size",height:Math.ceil(h+16)},"*")}window.addEventListener("load",function(){setTimeout(fit,0)});window.addEventListener("resize",fit)})();\x3C/script>\x3Cscript defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">\x3C/script>\x3Cstyle>html,body{margin:0;background:#eee;overflow:hidden}.page{position:relative;width:${pageWidth}px;min-height:${maxY}px;margin:8px auto;background:#fff;font:${pageFontSize}px/1.45 Arial,"Microsoft YaHei",sans-serif}.ocr-main-title{font-size:1.35em;font-weight:700}\x3C/style>\x3Cdiv class="page">${items}\x3C/div>`
}

// Chunker 为保证检索上下文会保留 overlap；预览按页合并时去掉相邻
// chunk 的重复前后缀，避免表格尾部/注释在页面中重复显示。
function mergeChunkOverlap(chunks) {
  let merged = ''
  for (const raw of chunks || []) {
    const text = String(raw || '').trim()
    if (!text) continue
    if (!merged) { merged = text; continue }
    const max = Math.min(2000, merged.length, text.length)
    let overlap = 0
    for (let size = max; size >= 24; size -= 1) {
      if (merged.slice(-size) === text.slice(0, size)) {
        overlap = size
        break
      }
    }
    merged += '\n\n' + text.slice(overlap)
  }
  // 同一 chunk 内也可能包含 pipeline 重复输出；按规范化文本指纹
  // 去掉重复段落/表格片段，但不改变不同表格行的顺序。
  const seen = new Set()
  const parts = merged.split(/\n\s*\n/)
  const unique = []
  for (const part of parts) {
    const key = part.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase()
    if (!key || seen.has(key)) continue
    seen.add(key)
    unique.push(part)
  }
  // 标题/注释有时被 pipeline 作为独立行重复返回，跨 HTML 片段也能去重。
  const seenLines = new Set()
  const lines = unique.join('\n\n').split('\n')
  return lines.filter(line => {
    const key = line.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase()
    const semantic = key.length >= 8 && /[\u4e00-\u9fff]/.test(key)
    if (!semantic) return true
    if (seenLines.has(key)) return false
    seenLines.add(key)
    return true
  }).join('\n')
}

// OCR 内容可能包含表格 HTML，但仍属于不可信输入。只允许被动排版标签，
// 删除脚本、外部资源、事件属性和 javascript: URL 后再交给预览 DOM。
function renderPreviewHtml(raw) {
  const source = String(raw || '')
  const template = document.createElement('template')
  template.innerHTML = source
  template.content.querySelectorAll('script, iframe, object, embed, link, style, form').forEach(el => el.remove())
  template.content.querySelectorAll('*').forEach(el => {
    for (const attr of [...el.attributes]) {
      const name = attr.name.toLowerCase()
      const value = attr.value.trim().toLowerCase()
      if (name.startsWith('on') || ['src', 'href', 'action', 'formaction'].includes(name) || value.startsWith('javascript:')) {
        el.removeAttribute(attr.name)
      }
    }
  })
  return template.innerHTML
    .replace(/\n/g, '<br>')
    .replace(/(^|<br>)###?\s+([^<]+)/g, '$1<strong>$2</strong>')
}

// ── Edit mode ──
const isEditing = ref(false)
const editLoading = ref(false)
const saving = ref(false)
const pendingEdits = reactive(new Map())
const displayLayoutPages = computed(() => (
  isEditing.value ? editableLayoutPages.value : cachedLayoutPages.value
))
const editRecords = computed(() => [...pendingEdits.values()])

async function startEdit() {
  if (!layoutRevision.value || !cachedLayoutPages.value.length) {
    ElMessage.warning('当前文件没有可编辑的版面缓存，请先重新解析')
    return
  }
  editLoading.value = true
  try {
    const data = await API.getFileContent(props.fileHash, true)
    const pages = Array.isArray(data.layout_pages) ? data.layout_pages : []
    if (!data.layout_revision || !pages.length) throw new Error('后端未返回可编辑版面')
    editableLayoutPages.value = pages
    layoutRevision.value = data.layout_revision
    pendingEdits.clear()
    isEditing.value = true
    showOcrCompare.value = false
    rebuildCatalog(data.outline)
    await nextTick()
  } catch (e) {
    ElMessage.error(`进入编辑失败：${e.message || e}`)
  } finally {
    editLoading.value = false
  }
}

function cancelEdit() {
  pendingEdits.clear()
  editableLayoutPages.value = []
  isEditing.value = false
}

async function saveEdit() {
  if (!pendingEdits.size) {
    ElMessage.info('没有需要保存的修改')
    return
  }
  saving.value = true
  try {
    const edits = [...pendingEdits.values()].map(item => ({
      page_num: item.page_num,
      block_index: item.block_index,
      op: item.op,
      content: item.content,
      content_format: item.content_format,
    }))
    const result = await API.saveFileContent(
      props.fileHash,
      layoutRevision.value,
      edits,
    )
    pendingEdits.clear()
    editableLayoutPages.value = []
    isEditing.value = false
    await loadFileContent()
    ElMessage.success(`已保存并同步 ${result.chunks_created} 个向量块`)
  } catch (e) {
    const detail = e?.body?.detail || e.message || e
    ElMessage.error(`保存失败：${detail}`)
  } finally {
    saving.value = false
  }
}

const recordsCollapsed = ref(false)

function restoreRecord(record) {
  const frame = [...(contentRef.value?.querySelectorAll('.dp-primary-layout__frame[data-page-num]') || [])]
    .find(item => Number(item.dataset.pageNum) === Number(record.page_num))
  frame?.contentWindow?.postMessage({
    type: 'rongneng-layout-edit-restore',
    block_index: record.block_index,
    op: record.op,
    content: record.before,
    content_format: record.content_format,
  }, '*')
  pendingEdits.delete(record.id)
  ElMessage.info('已恢复该处修改')
}
</script>

<style scoped>
.dp-layout { display: flex; flex-direction: column; height: 100vh; background: #f5f6f8; font-size: 14px; color: #333; }

/* Toolbar */
.dp-toolbar {
  height: 56px; flex-shrink: 0; background: #fff;
  border-bottom: 1px solid #ebeef0;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 24px;
}
.dp-toolbar__left-btn { border-color: #dcdfe6; color: #606266; }
.dp-toolbar__center { display: flex; align-items: center; gap: 12px; }
.dp-toolbar__title { font-size: 15px; font-weight: 600; color: #303133; max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dp-toolbar__docnum { font-size: 12px; color: #909399; background: #f0f0f0; padding: 2px 8px; border-radius: 3px; }
.dp-toolbar__editing { font-size: 12px; font-weight: 600; color: #1f5fbf; background: #eaf2ff; border: 1px solid #c9dcfb; padding: 3px 8px; border-radius: 3px; white-space: nowrap; }
.dp-toolbar__right-btn { min-width: 88px; }
.dp-toolbar__actions { display: flex; gap: 8px; align-items: center; }

/* Body */
.dp-body { flex: 1; display: flex; overflow: hidden; }

/* Left nav */
.dp-nav {
  width: 260px; flex-shrink: 0; background: #fff;
  border-right: 1px solid #ebeef0; padding: 20px; overflow-y: auto;
}
.dp-nav__title { font-weight: 700; font-size: 16px; margin-bottom: 16px; }
.dp-nav__loading, .dp-nav__empty { color: #909399; font-size: 13px; padding: 12px 0; }
.dp-nav__item { display: flex; align-items: flex-start; gap: 6px; padding: 8px 8px 8px 8px; color: #3a8ee6; cursor: pointer; line-height: 1.45; font-size: 13px; border-radius: 3px; }
.dp-nav__item-text { flex: 1; min-width: 0; word-break: break-word; }
.dp-nav__item-page { flex: 0 0 auto; color: #b1b5bd; font-size: 11px; line-height: 1.5; }
.dp-nav__item.is-root { font-weight: 700; color: #303133; }
.dp-nav__item.is-active { color: #e6a23c; font-weight: 600; }
.dp-nav__item:hover { opacity: 0.8; }

/* Center content */
.dp-content { flex: 1; min-width: 0; overflow-y: auto; overflow-x: hidden; padding: 24px; background: #f5f6f8; }
.dp-content__paper {
  background: #fff; border-radius: 4px;
  padding: 24px 30px; outline: none;
  font-size: 15px; color: #1f2329; line-height: 1.8;
  white-space: pre-wrap; word-break: break-all;
}
.dp-content__paper.is-editing {
  min-height: 100%; padding: 60px 50px;
  box-shadow: 0 0 0 2px #409eff inset; cursor: text;
}
/* 分页块: 每页一页纸, 页间留白 */
.dp-page { margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.dp-page__title { font-size: 15px; font-weight: 700; color: #909399; margin-bottom: 12px; }
.dp-page__body { font-size: 15px; color: #1f2329; line-height: 1.8; white-space: pre-wrap; word-break: break-all; }
.dp-page__body :deep(table) { width: 100%; margin: 12px 0; border-collapse: collapse; table-layout: auto; background: #fff; }
.dp-page__body :deep(td), .dp-page__body :deep(th) { border: 1px solid #8c939d; padding: 5px 7px; text-align: center; vertical-align: middle; line-height: 1.45; white-space: normal; word-break: normal; }
.dp-page__body :deep(th) { background: #f5f7fa; font-weight: 600; }
.dp-ocr-compare { margin-bottom: 20px; background: #fff; border-radius: 4px; padding: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.dp-ocr-compare__head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.dp-ocr-compare__grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.dp-ocr-card { min-width: 0; border: 1px solid #ebeef5; border-radius: 4px; overflow: hidden; }
.dp-ocr-card h3 { margin: 0; padding: 10px 12px; background: #f5f7fa; font-size: 14px; }
.dp-ocr-card pre { margin: 0; padding: 12px; max-height: none; overflow: visible; white-space: pre-wrap; word-break: break-word; font: 13px/1.6 Consolas, monospace; }
.dp-ocr-pages { max-height: none; overflow: visible; }
.dp-ocr-page { border-top: 1px solid #ebeef5; }
.dp-ocr-page:first-child { border-top: 0; }
.dp-ocr-page h4 { margin: 0; padding: 9px 12px; background: #fafafa; color: #606266; font-size: 13px; }
.dp-ocr-frame { display: block; width: 100%; max-width: 960px; height: 520px; margin: 0 auto; border: 0; background: #fff; overflow: hidden; }
.dp-primary-layout { width: 100%; margin: 0 auto 8px; background: #f3f4f6; }
.dp-primary-layout__page { width: 100%; margin: 0 auto 8px; background: #f3f4f6; }
.dp-primary-layout__page:last-child { margin-bottom: 0; }
.dp-primary-layout__title { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 3px 8px; color: #909399; font-size: 12px; background: transparent; }
.dp-primary-layout__hint { color: #2f6fdb; font-weight: 600; }
.dp-primary-layout__page.is-editing { box-shadow: inset 3px 0 0 #2f6fdb; }
.dp-primary-layout__frame { display: block; width: 100%; max-width: 960px; height: 900px; margin: 0 auto; border: 0; background: #f3f4f6; overflow: hidden; }
.dp-ocr-empty { padding: 24px; color: #909399; }
.dp-edit-unavailable { max-width: 760px; margin: 40px auto; color: #606266; text-align: center; }
@media (max-width: 1100px) { .dp-ocr-compare__grid { grid-template-columns: 1fr; } }

/* Right sidebar */
.dp-side {
  width: 380px; flex-shrink: 0; background: #fff;
  border-left: 1px solid #ebeef0; padding: 20px; overflow-y: auto;
}
.dp-side__header {
  display: flex; align-items: center; justify-content: space-between;
  font-weight: 700; font-size: 15px; cursor: pointer;
  padding-bottom: 16px; border-bottom: 1px solid #ebeef0; margin-bottom: 16px;
}
.dp-side__count { color: #f56c6c; }
.dp-side__toggle { transition: transform 0.2s; }
.dp-side__toggle.is-collapsed { transform: rotate(180deg); }
.dp-side__empty { color: #909399; text-align: center; padding: 24px 0; }

.dp-record {
  border: 1px solid #ebeef0; border-radius: 6px;
  padding: 16px; margin-bottom: 16px;
}
.dp-record__header { display: flex; align-items: center; justify-content: space-between; font-weight: 700; margin-bottom: 10px; }
.dp-record__restore { color: #3a8ee6; font-weight: 400; font-size: 13px; cursor: pointer; }
.dp-record__label { color: #606266; margin: 8px 0 4px; }
.dp-record__before { color: #f56c6c; text-decoration: line-through; line-height: 1.6; word-break: break-all; white-space: pre-wrap; }
.dp-record__after { color: #303133; line-height: 1.6; word-break: break-all; white-space: pre-wrap; }
.dp-record__before, .dp-record__after { max-height: 160px; overflow: auto; }
</style>
