<template>
  <div class="excel-workbench">
    <!-- Left: workbook list -->
    <div class="wb-list-panel">
      <div class="wb-list-header">
        <span class="wb-list-title">📚 我的 Excel 库</span>
        <el-button type="primary" size="small" @click="uploadVisible = true">＋ 上传 Excel</el-button>
      </div>

      <div v-if="loadingList" class="wb-empty">加载中…</div>
      <div v-else-if="!workbooks.length" class="wb-empty">
        <p>还没有工作簿</p>
        <p class="wb-hint">上传 Excel 文件,审核确认后即可进行 SQL 问答与报告分析。</p>
      </div>

      <div
        v-for="wb in workbooks"
        :key="wb.import_id"
        :class="['wb-item', { active: selected?.import_id === wb.import_id }]"
        @click="selectWorkbook(wb)"
      >
        <div class="wb-item-top">
          <span class="wb-name">{{ wb.file_name }}</span>
          <el-tag size="small" :type="wbTagType(wb.status)">{{ wbStatusLabel(wb.status) }}</el-tag>
        </div>
        <div class="wb-item-meta">
          <span>v{{ wb.version }}</span>
          <span v-if="wb.sheets?.length">{{ wb.sheets.length }} 个 Sheet</span>
          <span v-if="wb.sheets?.length">{{ dataSheetCount(wb) }} 表</span>
        </div>
      </div>
    </div>

    <!-- Right: three states -->
    <div class="wb-main">
      <el-empty v-if="!selected" description="从左侧选择工作簿,或上传新的 Excel 文件" />

      <!-- Review state -->
      <ExcelDraftReview
        v-else-if="selected.status === 'draft' || selected.status === 'ready'"
        :import-data="selected"
        @cancel="selected = null"
        @activated="onActivated"
      />

      <!-- QA state -->
      <ExcelQA v-else-if="selected.status === 'active'" :workbook="selected" />

      <!-- Other states -->
      <div v-else class="wb-state-hint">
        <p>{{ selected.file_name }} — 状态: {{ wbStatusLabel(selected.status) }}</p>
        <p v-if="selected.status === 'failed'" class="wb-err">{{ selected.error_message }}</p>
        <el-button v-if="selected.status === 'failed'" size="small" @click="reload">重新加载</el-button>
      </div>
    </div>

    <!-- Upload dialog -->
    <el-dialog v-model="uploadVisible" title="上传 Excel 工作簿" width="520px" destroy-on-close>
      <el-upload
        drag
        :auto-upload="false"
        :limit="1"
        accept=".xlsx,.xlsm,.csv"
        :on-change="onFileChange"
        :on-remove="() => (pendingFile = null)"
      >
        <el-icon :size="40" color="#909399"><UploadFilled /></el-icon>
        <div class="el-upload__text">将 Excel 文件拖到此处,或<em>点击上传</em></div>
        <template #tip>
          <div class="el-upload__tip">支持 .xlsx / .xlsm / .csv</div>
        </template>
      </el-upload>

      <template #footer>
        <el-button @click="uploadVisible = false">取消</el-button>
        <el-button type="primary" :loading="uploading" :disabled="!pendingFile" @click="doUpload">
          {{ uploading ? '解析中…' : '上传并生成审核草稿' }}
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { UploadFilled } from '@element-plus/icons-vue'
import { createWorkbookImport, getWorkbookImport, listWorkbookImports } from '../api.js'
import ExcelDraftReview from './ExcelDraftReview.vue'
import ExcelQA from './ExcelQA.vue'

const workbooks = ref([])
const selected = ref(null)
const loadingList = ref(false)
const uploadVisible = ref(false)
const uploading = ref(false)
const pendingFile = ref(null)
const ACTIVE_WORKBOOK_KEY = 'excel-active-workbook'

function rememberWorkbook(wb) {
  if (!wb?.import_id) return
  try {
    localStorage.setItem(ACTIVE_WORKBOOK_KEY, JSON.stringify({
      import_id: wb.import_id,
      workbook_id: wb.workbook_id || null,
    }))
  } catch (_) {
    // localStorage may be unavailable in private/embedded contexts.
  }
}

function rememberedWorkbook() {
  try {
    const raw = localStorage.getItem(ACTIVE_WORKBOOK_KEY)
    return raw ? JSON.parse(raw) : null
  } catch (_) {
    return null
  }
}

async function refreshList() {
  loadingList.value = true
  try {
    workbooks.value = await listWorkbookImports({})
    const currentId = selected.value?.import_id
    const saved = rememberedWorkbook()
    const targetId = currentId || saved?.import_id
    const restored = workbooks.value.find((wb) => wb.import_id === targetId)
    if (restored) {
      selected.value = restored
      rememberWorkbook(restored)
    }
  } catch (e) {
    workbooks.value = []
  } finally {
    loadingList.value = false
  }
}

function selectWorkbook(wb) {
  selected.value = wb
  rememberWorkbook(wb)
}

function onFileChange(file) {
  pendingFile.value = file.raw
}

async function doUpload() {
  if (!pendingFile.value) return
  uploading.value = true
  try {
    const data = await createWorkbookImport(pendingFile.value, 'agent')
    await refreshList()
    const fresh = await getWorkbookImport(data.import_id)
    selected.value = fresh
    rememberWorkbook(fresh)
    uploadVisible.value = false
  } catch (e) {
    window.alert('上传失败: ' + (e?.message || e))
  } finally {
    uploading.value = false
  }
}

async function onActivated(confirm) {
  await refreshList()
  const data = await getWorkbookImport(confirm.import_id)
  selected.value = { ...data, workbook_id: confirm.workbook_id, active_version: confirm.version }
  rememberWorkbook(selected.value)
}

async function reload() {
  if (selected.value) {
    selected.value = await getWorkbookImport(selected.value.import_id)
    rememberWorkbook(selected.value)
  }
}

function wbStatusLabel(s) {
  return { ready: '可建库', draft: '需审核', active: '已建库', failed: '失败', superseded: '历史版本', parsing: '解析中' }[s] || s
}
function wbTagType(s) {
  return { ready: 'success', active: 'success', draft: 'warning', failed: 'danger', superseded: 'info', parsing: 'info' }[s] || 'info'
}
function dataSheetCount(wb) {
  return (wb.sheets || []).filter((s) => s.kind === 'data' && s.enabled !== false).length
}

onMounted(refreshList)
</script>

<style scoped>
.excel-workbench { display: flex; gap: 12px; height: 100%; padding: 12px; }
.wb-list-panel {
  width: 280px; flex-shrink: 0; background: #fff; border-radius: 8px;
  display: flex; flex-direction: column; overflow: hidden;
}
.wb-list-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px; border-bottom: 1px solid #f0f2f5;
}
.wb-list-title { font-weight: 600; font-size: 14px; }
.wb-empty { padding: 32px 16px; text-align: center; color: #909399; font-size: 13px; }
.wb-hint { font-size: 12px; color: #b0b3b8; margin-top: 8px; line-height: 1.6; }
.wb-item { padding: 10px 12px; border-bottom: 1px solid #f5f6f8; cursor: pointer; }
.wb-item:hover { background: #f8faf9; }
.wb-item.active { background: #e8f7f5; border-left: 3px solid #0f9c8f; }
.wb-item-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.wb-name { font-size: 13px; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.wb-item-meta { display: flex; gap: 10px; color: #909399; font-size: 12px; margin-top: 4px; }
.wb-main { flex: 1; min-width: 0; overflow: auto; }
.wb-state-hint { text-align: center; padding: 60px 0; color: #606266; }
.wb-err { color: #f56c6c; margin-top: 8px; }
</style>
