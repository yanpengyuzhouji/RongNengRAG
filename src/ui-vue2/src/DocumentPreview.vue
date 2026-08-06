<template>
  <div class="dp-layout">
    <!-- Top toolbar -->
    <header class="dp-toolbar">
      <el-button class="dp-toolbar__left-btn" @click="isEditing ? cancelEdit() : $emit('close')">
        {{ isEditing ? '取消编辑' : '关闭' }}
      </el-button>
      <div class="dp-toolbar__center">
        <span class="dp-toolbar__title">{{ fileData.file_name }}</span>
        <span v-if="fileData.doc_number" class="dp-toolbar__docnum">{{ fileData.doc_number }}</span>
      </div>
      <el-button type="primary" plain class="dp-toolbar__right-btn" @click="isEditing ? saveEdit() : startEdit()">
        {{ isEditing ? '保存' : '编辑' }}
      </el-button>
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
            :class="['dp-nav__item', { 'is-active': activeCatalog === item.id }]"
            @click="scrollTo(item.id)"
          >
            {{ idx + 1 }}、{{ item.title }}
          </div>
          <div v-if="!catalog.length" class="dp-nav__empty">暂无目录</div>
        </template>
      </aside>

      <!-- Center: document content (按页渲染, 目录锚点跳转) -->
      <main class="dp-content" ref="contentRef">
        <div v-if="loading" class="dp-content__paper" style="color:#909399">加载中...</div>
        <div v-else-if="error" class="dp-content__paper" style="color:#f56c6c">{{ error }}</div>
        <template v-else-if="!isEditing">
          <div
            v-for="block in pageBlocks"
            :key="block.page"
            :id="`page-${block.page}`"
            class="dp-content__paper dp-page"
          >
            <div class="dp-page__title">## 第 {{ block.page }} 页</div>
            <div class="dp-page__body" v-text="block.text"></div>
          </div>
        </template>
        <div
          v-else
          class="dp-content__paper is-editing"
          contenteditable
          ref="paperRef"
          @input="onContentInput"
          v-text="docContent"
        ></div>
      </main>

      <!-- Right: edit history -->
      <aside class="dp-side">
        <div class="dp-side__header" @click="recordsCollapsed = !recordsCollapsed">
          <span>本文档共计 <span class="dp-side__count">{{ editRecords.length }}</span> 处编辑记录</span>
          <el-icon class="dp-side__toggle" :class="{ 'is-collapsed': recordsCollapsed }">
            <ArrowUp />
          </el-icon>
        </div>

        <div v-show="!recordsCollapsed" class="dp-side__list">
          <div v-if="editRecords.length === 0" class="dp-side__empty">暂无编辑记录</div>

          <div v-for="(record, idx) in editRecords" :key="record.id" class="dp-record">
            <div class="dp-record__header">
              <span class="dp-record__title">编辑记录{{ idx + 1 }}</span>
              <span v-if="isEditing" class="dp-record__restore" @click="restoreRecord(record)">恢复</span>
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
import { ref, reactive, computed, onMounted, nextTick } from 'vue'
import { ArrowUp } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import * as API from './api.js'

const props = defineProps({
  fileHash: { type: String, required: true },
  fileName: { type: String, default: '' },
})

defineEmits(['close'])

// 与 api.js 共用同一后端地址配置
const BASE = (import.meta.env?.VITE_API_BASE || 'http://localhost:8000').replace(/\/$/, '')

// ── Fetch file content ──
const loading = ref(true)
const error = ref('')
const fileData = reactive({
  file_name: props.fileName,
  doc_number: '',
  domain: '',
  category: '',
  full_text: '',
  chunks: [],
  total_chunks: 0,
  total_pages: 0,
})

onMounted(async () => {
  try {
    const resp = await fetch(`${BASE}/files/${encodeURIComponent(props.fileHash)}/content`)
    if (!resp.ok) { error.value = `加载失败: HTTP ${resp.status}`; loading.value = false; return }
    const data = await resp.json()
    Object.assign(fileData, data)
    docContent.value = data.full_text || '（无文本内容）'
    // 按页聚合 chunk → 渲染为分页块
    pageBlocks.splice(0, pageBlocks.length)
    const pageMap = new Map()
    for (const c of (data.chunks || [])) {
      const p = c.page_num || 0
      if (!pageMap.has(p)) pageMap.set(p, { page: p, text: [] })
      pageMap.get(p).text.push(c.text)
    }
    for (const { page, text } of pageMap.values()) {
      pageBlocks.push({ page, text: text.join('\n\n') })
    }
    // 目录: 按页码排序
    catalog.splice(0, catalog.length,
      ...pageBlocks.map(b => ({ id: `page-${b.page}`, title: `第 ${b.page} 页`, page: b.page })))
    if (catalog.length > 0) activeCatalog.value = catalog[0].id
    loading.value = false
  } catch (e) {
    error.value = '加载失败: ' + (e.message || e)
    loading.value = false
  }
})

// ── Catalog ──
const pageBlocks = reactive([])     // 分页渲染块
const catalog = reactive([])
const activeCatalog = ref('')
const contentRef = ref(null)

function scrollTo(id) {
  activeCatalog.value = id
  nextTick(() => {
    const el = contentRef.value?.querySelector(`#${id}`)
    el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  })
}

// ── Edit mode ──
const docContent = ref('')
const originalContent = ref('')
const isEditing = ref(false)
const paperRef = ref(null)

function onContentInput(e) {
  docContent.value = e.target.innerText
}

function startEdit() {
  isEditing.value = true
  originalContent.value = docContent.value
  nextTick(() => paperRef.value?.focus())
}

function cancelEdit() {
  docContent.value = originalContent.value
  isEditing.value = false
}

function saveEdit() {
  if (docContent.value !== originalContent.value) {
    editRecords.value.push({
      id: Date.now(),
      before: originalContent.value,
      after: docContent.value,
    })
  }
  originalContent.value = docContent.value
  isEditing.value = false
  ElMessage.success('保存成功')
}

// ── Edit records (in-memory only, ponytail: no backend persistence) ──
const editRecords = ref([])
const recordsCollapsed = ref(false)

function restoreRecord(record) {
  docContent.value = record.before
  editRecords.value = editRecords.value.filter(r => r.id !== record.id)
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
.dp-toolbar__right-btn { min-width: 88px; }

/* Body */
.dp-body { flex: 1; display: flex; overflow: hidden; }

/* Left nav */
.dp-nav {
  width: 260px; flex-shrink: 0; background: #fff;
  border-right: 1px solid #ebeef0; padding: 20px; overflow-y: auto;
}
.dp-nav__title { font-weight: 700; font-size: 16px; margin-bottom: 16px; }
.dp-nav__loading, .dp-nav__empty { color: #909399; font-size: 13px; padding: 12px 0; }
.dp-nav__item { padding: 8px 0; color: #3a8ee6; cursor: pointer; line-height: 1.6; font-size: 13px; }
.dp-nav__item.is-active { color: #e6a23c; font-weight: 600; }
.dp-nav__item:hover { opacity: 0.8; }

/* Center content */
.dp-content { flex: 1; overflow-y: auto; padding: 24px; background: #f5f6f8; }
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
</style>
