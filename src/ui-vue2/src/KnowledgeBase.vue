<template>
  <div class="kb-layout">
    <!-- Sidebar -->
    <aside class="kb-sidebar">
      <div class="sidebar-title">知识库</div>
      <div class="sidebar-action" @click="openDirDialog()">
        <el-icon><Plus /></el-icon>
        <span>创建知识目录</span>
      </div>
    </aside>

    <!-- Main -->
    <main class="kb-main">
      <!-- Stats bar -->
      <div class="stats-bar" v-if="summary">
        <span class="stat-item">📄 {{ summary.total_files }} 文件</span>
        <span class="stat-item">📦 {{ summary.total_chunks }} 片段</span>
        <span v-if="missingCount" class="stat-item stat-warn">⚠ {{ missingCount }} 丢失</span>
      </div>

      <!-- Filters -->
      <div class="filter-row">
        <el-input v-model="searchKw" placeholder="搜索文件名或文档编号" clearable class="search-input" @clear="fetchData" @keyup.enter="fetchData">
          <template #prefix><el-icon><Search /></el-icon></template>
        </el-input>
        <el-select v-model="filterStatus" placeholder="解析状态" clearable @change="fetchData" style="width:130px">
          <el-option label="已完成" value="completed" />
          <el-option label="处理中" value="pending" />
          <el-option label="失败" value="failed" />
        </el-select>
        <el-select v-model="filterDomain" placeholder="专业域" clearable @change="onDomainChange" style="width:120px">
          <el-option v-for="d in domainOptions" :key="d" :label="d" :value="d" />
        </el-select>
        <el-select v-model="filterCategory" placeholder="类目" clearable @change="fetchData" style="width:140px">
          <el-option v-for="c in categoryOptions" :key="c" :label="c" :value="c" />
        </el-select>
        <el-button type="primary" @click="fetchData">查询</el-button>
        <el-button @click="resetFilters">重置</el-button>
        <el-button type="success" @click="uploadVisible = true">上传</el-button>
      </div>

      <!-- Table -->
      <el-table :data="tableData" stripe style="width: 100%" v-loading="loading">
        <el-table-column label="文档名称" min-width="280">
          <template #default="{ row }">
            <div class="file-name-cell">
              <img :src="fileIcon(row.file_type)" class="file-type-icon" />
              <span class="file-name-text" :title="row.file_name">{{ row.file_name }}</span>
              <span v-if="row.doc_number" class="doc-number-badge">{{ row.doc_number }}</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="知识目录" width="180">
          <template #default="{ row }">
            <span class="catalog-text">{{ row.domain }}{{ row.category ? ' / ' + row.category : '' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="类型" width="70">
          <template #default="{ row }">{{ (row.file_type || '').replace('.', '') }}</template>
        </el-table-column>
        <el-table-column label="解析状态" width="110">
          <template #default="{ row }">
            <div class="status-cell">
              <span :class="['status-dot', statusClass(row.status)]"></span>
              <span class="status-text">{{ statusText(row.status) }}</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="片段" width="70">
          <template #default="{ row }">{{ row.chunks_count }}</template>
        </el-table-column>
        <el-table-column label="更新时间" width="160">
          <template #default="{ row }">{{ (row.updated_at || row.created_at || '').slice(0, 16) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="240" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" size="small" @click="openEdit(row)">编辑</el-button>
            <el-button link type="primary" size="small" @click="API.downloadFile(row.file_hash)">下载</el-button>
            <el-button link type="success" size="small" @click="$emit('preview', row)">预览</el-button>
            <el-button v-if="row.status === 'failed'" link type="warning" size="small" @click="handleReindex(row)">重解析</el-button>
            <el-button v-if="row.status !== 'pending'" link type="danger" size="small" @click="handleDelete(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>

      <!-- Pagination -->
      <div class="pagination-wrap">
        <el-pagination
          v-model:current-page="page.current"
          :page-size="page.size"
          :page-sizes="[10, 20, 50]"
          :total="page.total"
          layout="total, prev, pager, next, sizes, jumper"
          @size-change="onPageSizeChange"
          @current-change="onPageChange"
        />
      </div>
    </main>

    <!-- FAB: AI Assistant -->
    <div class="kb-fab" @click="$emit('goAi')">
      <div class="fab-icon">🤖</div>
      <div class="fab-label">AI 小助手</div>
    </div>

    <!-- ════════ Upload Dialog ════════ -->
    <el-dialog v-model="uploadVisible" title="上传文件" width="640px" destroy-on-close>
      <div class="upload-area" @click="triggerFileInput" v-if="!uploadFiles.length">
        <el-icon :size="40" color="#909399"><UploadFilled /></el-icon>
        <p style="margin-top:10px;color:#606266">点击选择文件或拖拽到此处</p>
        <p style="font-size:12px;color:#c0c4cc">支持 PDF / Word / Excel / PPT / OFD / WPS / TXT</p>
      </div>
      <div v-else class="upload-file-list">
        <div v-for="(f, i) in uploadFiles" :key="i" class="upload-file-item">
          <span class="uf-name">{{ f.file.name }}</span>
          <span v-if="f.status === 'pending'" class="uf-status pending">等待中</span>
          <span v-else-if="f.status === 'uploading'" class="uf-status running">处理中</span>
          <span v-else-if="f.status === 'done'" class="uf-status done">✅ 完成 ({{ f.result?.chunks_created || 0 }} 片段)</span>
          <span v-else-if="f.status === 'error'" class="uf-status failed">❌ {{ f.error }}</span>
          <el-progress v-if="f.status === 'uploading'" :percentage="f.progress" :show-text="false" style="width:120px" />
          <el-button v-if="f.status === 'error'" link type="danger" size="small" @click="retryUpload(i)">重试</el-button>
        </div>
      </div>
      <input ref="fileInputRef" type="file" multiple hidden
        accept=".pdf,.doc,.docx,.xls,.xlsx,.ppt,.pptx,.wps,.ofd,.txt,.md,.jpg,.jpeg,.png"
        @change="onFilesSelected" />

      <!-- Directory selectors -->
      <div class="dir-selectors" v-if="uploadFiles.length">
        <span class="dir-label">选择知识目录（选填）</span>
        <div class="cascade-row">
          <el-select v-model="uploadDomain" placeholder="一级: 专业域" clearable @change="onUploadDomainChange" style="width:130px">
            <el-option v-for="d in domainOptions" :key="d" :label="d" :value="d" />
          </el-select>
          <el-select v-model="uploadCategory" placeholder="二级: 类目" clearable @change="onUploadCategoryChange" style="width:150px">
            <el-option v-for="c in uploadCategoryOptions" :key="c" :label="c" :value="c" />
          </el-select>
          <el-select v-model="uploadSubcategory" placeholder="三级: 子类目" clearable style="width:150px">
            <el-option v-for="s in uploadSubcategoryOptions" :key="s" :label="s" :value="s" />
          </el-select>
          <el-select v-model="uploadFileType" placeholder="四级: 文件类型" clearable style="width:120px">
            <el-option label="PDF" value=".pdf" />
            <el-option label="Word" value=".docx" />
            <el-option label="Excel" value=".xlsx" />
            <el-option label="PPT" value=".pptx" />
            <el-option label="OFD" value=".ofd" />
            <el-option label="图片" value="image" />
          </el-select>
        </div>
      </div>

      <template #footer>
        <el-button @click="uploadVisible = false">关闭</el-button>
        <el-button type="primary" @click="startUpload" :disabled="!uploadFiles.length || uploading">
          {{ uploading ? '上传中...' : '开始上传' }}
        </el-button>
      </template>
    </el-dialog>

    <!-- ════════ Edit Dialog ════════ -->
    <el-dialog v-model="editVisible" title="编辑文件信息" width="500px" destroy-on-close>
      <el-form :model="editForm" label-width="100px">
        <el-form-item label="文件名">
          <el-input :model-value="editForm.file_name" disabled />
        </el-form-item>
        <el-form-item label="专业域">
          <el-select v-model="editForm.domain" style="width:100%" @change="onEditDomainChange">
            <el-option v-for="d in domainOptions" :key="d" :label="d" :value="d" />
          </el-select>
        </el-form-item>
        <el-form-item label="类目">
          <el-select v-model="editForm.category" style="width:100%">
            <el-option v-for="c in editCategoryOptions" :key="c" :label="c" :value="c" />
          </el-select>
        </el-form-item>
        <el-form-item label="文档编号">
          <el-input v-model="editForm.doc_number" placeholder="如 GB50052-2009" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="editVisible = false">取消</el-button>
        <el-button type="primary" @click="submitEdit" :loading="editSaving">保存</el-button>
      </template>
    </el-dialog>

    <!-- ════════ Create Directory Dialog (shell) ════════ -->
    <el-dialog v-model="dirVisible" title="新建知识目录" width="560px" align-center destroy-on-close>
      <el-form ref="dirFormRef" :model="dirForm" :rules="dirRules" label-width="100px">
        <el-form-item label="目录名称" prop="name">
          <el-input v-model="dirForm.name" placeholder="请输入目录名称" />
        </el-form-item>
        <el-form-item label="上级目录" prop="parent">
          <el-select v-model="dirForm.parent" placeholder="请选择上级目录" style="width:100%">
            <el-option label="配电知识库" value="配电知识库" />
            <el-option label="变电知识库" value="变电知识库" />
            <el-option label="送电输电知识库" value="送电输电知识库" />
            <el-option label="综合知识库" value="综合知识库" />
          </el-select>
        </el-form-item>
        <el-form-item label="排序号" prop="sort">
          <el-input-number v-model="dirForm.sort" :min="0" controls-position="right" style="width:100%" />
        </el-form-item>
        <el-form-item label="说明">
          <el-input v-model="dirForm.desc" type="textarea" :rows="4" maxlength="100" show-word-limit placeholder="请输入说明（选填）" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dirVisible = false">取消</el-button>
        <el-button type="primary" @click="submitDirForm">确定</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import * as API from './api.js'

defineEmits(['goAi', 'preview'])

// ════════ Config ════════
const domainOptions = ['变电', '配电', '送电输电', '综合']
const allCategories = {
  变电: ['标准规范', '技术文件', '设计图纸', '设计规范', '通知意见', '会议纪要', '培训资料', '物资采购', '造价与费用', '三维设计', '研究技术报告', '设计公司文件'],
  配电: ['标准规范', '技术文件', '通用设计典设', '设计规范', '图审要求', '差异化设计', '通知意见', '物料与技术规范书', '造价与费用', '培训资料', '管理办法', '设计图集', '专题场景'],
  送电输电: ['标准规范', '设计规程', '反事故措施', '水土保持环保', '通信', '通知意见'],
  综合: ['省公司发文', '地市公司发文', '公共培训资料', '跨域通用文件', '杂项'],
}

// ════════ State ════════
const tableData = ref([])
const loading = ref(false)
const summary = ref(null)
const missingCount = ref(0)
const page = reactive({ current: 1, size: 10, total: 0 })
const searchKw = ref('')
const filterStatus = ref('')
const filterDomain = ref('')
const filterCategory = ref('')
const categoryOptions = ref([])

// ════════ Data fetching ════════
async function fetchData() {
  loading.value = true
  const offset = (page.current - 1) * page.size
  const res = await API.listFiles({
    limit: page.size,
    offset,
    status: filterStatus.value || null,
    domain: filterDomain.value || null,
  })
  if (typeof res === 'string') {
    ElMessage.error('加载失败: ' + res)
    loading.value = false
    return
  }
  let files = (res.files || []).filter(f => (f.file_name || '').toLowerCase() !== 'thumb.db')
  // Client-side keyword search (searches file_name + doc_number)
  // ponytail: no backend fulltext search, client-side filter for now
  if (searchKw.value) {
    const kw = searchKw.value.toLowerCase()
    files = files.filter(r =>
      (r.file_name || '').toLowerCase().includes(kw) ||
      (r.doc_number || '').toLowerCase().includes(kw)
    )
  }
  if (filterCategory.value) {
    files = files.filter(r => r.category === filterCategory.value)
  }
  tableData.value = files
  missingCount.value = res.missing_count || 0
  page.total = res.count || files.length
  loading.value = false
}

async function fetchSummary() {
  const res = await API.getFileSummary()
  if (typeof res !== 'string') summary.value = res
}

function onDomainChange() {
  if (filterDomain.value) {
    categoryOptions.value = allCategories[filterDomain.value] || []
  } else {
    categoryOptions.value = []
  }
  filterCategory.value = ''
  fetchData()
}

function onPageChange(p) { page.current = p; fetchData() }
function onPageSizeChange(s) { page.size = s; page.current = 1; fetchData() }

function resetFilters() {
  searchKw.value = ''
  filterStatus.value = ''
  filterDomain.value = ''
  filterCategory.value = ''
  categoryOptions.value = []
  page.current = 1
  fetchData()
}

// ════════ Upload ════════
const uploadVisible = ref(false)
const fileInputRef = ref(null)
const uploadFiles = ref([])
const uploading = ref(false)
const uploadDomain = ref('')
const uploadCategory = ref('')
const uploadSubcategory = ref('')
const uploadFileType = ref('')
const uploadCategoryOptions = ref([])
const uploadSubcategoryOptions = ref([])

function triggerFileInput() { fileInputRef.value?.click() }
function onFilesSelected(e) {
  const files = e.target.files
  if (!files?.length) return
  for (const f of files) {
    uploadFiles.value.push({ file: f, status: 'pending', progress: 0, result: null, error: '' })
  }
  fileInputRef.value.value = ''
}

function onUploadDomainChange() {
  uploadCategory.value = ''
  uploadSubcategory.value = ''
  uploadCategoryOptions.value = uploadDomain.value ? (allCategories[uploadDomain.value] || []) : []
  uploadSubcategoryOptions.value = []
}
function onUploadCategoryChange() {
  uploadSubcategory.value = ''
  if (uploadDomain.value && uploadCategory.value) {
    loadUploadSubcategories()
  } else {
    uploadSubcategoryOptions.value = []
  }
}
async function loadUploadSubcategories() {
  const res = await API.getSubcategories(uploadDomain.value, uploadCategory.value)
  if (typeof res !== 'string') {
    uploadSubcategoryOptions.value = [...new Set(res.map(r => r.category || '').filter(Boolean))]
  }
}

async function startUpload() {
  uploading.value = true
  const domain = uploadDomain.value || ''
  const category = uploadCategory.value || ''
  for (let i = 0; i < uploadFiles.value.length; i++) {
    const uf = uploadFiles.value[i]
    if (uf.status === 'done') continue
    uf.status = 'uploading'
    uf.progress = 20
    const res = await API.uploadFile(uf.file, domain, category)
    if (typeof res === 'string') {
      uf.status = 'error'
      uf.error = res
      uf.progress = 0
    } else if (res.success) {
      uf.status = 'done'
      uf.progress = 100
      uf.result = res
    } else {
      uf.status = 'error'
      uf.error = res.error_message || res.status
      uf.progress = 0
    }
  }
  uploading.value = false
  const done = uploadFiles.value.filter(f => f.status === 'done').length
  const fail = uploadFiles.value.filter(f => f.status === 'error').length
  if (done) ElMessage.success(`${done} 个文件入库完成`)
  if (fail) ElMessage.warning(`${fail} 个文件入库失败`)
  // Reset for next batch
  uploadFiles.value = []
  uploadDomain.value = ''
  uploadCategory.value = ''
  uploadSubcategory.value = ''
  uploadFileType.value = ''
  fetchData()
  fetchSummary()
}

function retryUpload(i) {
  const uf = uploadFiles.value[i]
  uf.status = 'pending'
  uf.error = ''
  uf.progress = 0
  startUpload()
}

// ════════ Row actions ════════
async function handleDelete(row) {
  try {
    await ElMessageBox.confirm(`确定删除 "${row.file_name}"？`, '确认删除', {
      confirmButtonText: '删除', cancelButtonText: '取消', type: 'warning',
    })
  } catch { return }
  const res = await API.deleteFile(row.file_hash, true)
  if (typeof res === 'string') { ElMessage.error('删除失败: ' + res); return }
  ElMessage.success('已删除')
  fetchData()
  fetchSummary()
}

async function handleReindex(row) {
  loading.value = true
  const res = await API.api('POST', `/files/${encodeURIComponent(row.file_hash)}/reindex`)
  loading.value = false
  if (typeof res === 'string') { ElMessage.error('重解析失败: ' + res); return }
  if (res.success) {
    ElMessage.success(`重解析完成: ${res.chunks_created} 片段`)
    fetchData()
    fetchSummary()
  } else {
    ElMessage.warning(res.status || '失败')
  }
}

// ════════ Edit ════════
const editVisible = ref(false)
const editSaving = ref(false)
const editForm = reactive({ file_name: '', domain: '', category: '', doc_number: '', file_hash: '' })
const editCategoryOptions = ref([])

function openEdit(row) {
  editForm.file_hash = row.file_hash
  editForm.file_name = row.file_name
  editForm.domain = row.domain || ''
  editForm.category = row.category || ''
  editForm.doc_number = row.doc_number || ''
  editCategoryOptions.value = row.domain ? (allCategories[row.domain] || []) : []
  editVisible.value = true
}
function onEditDomainChange() {
  editForm.category = ''
  editCategoryOptions.value = editForm.domain ? (allCategories[editForm.domain] || []) : []
}
async function submitEdit() {
  editSaving.value = true
  const res = await API.updateFileMeta(editForm.file_hash, {
    domain: editForm.domain,
    category: editForm.category,
    doc_number: editForm.doc_number,
  })
  editSaving.value = false
  if (typeof res === 'string') { ElMessage.error('保存失败: ' + res); return }
  ElMessage.success('已更新')
  editVisible.value = false
  fetchData()
}

// ════════ Helpers ════════
const iconMap = {
  '.pdf': 'https://cdn-icons-png.flaticon.com/128/337/337946.png',
  '.doc': 'https://cdn-icons-png.flaticon.com/128/281/281760.png',
  '.docx': 'https://cdn-icons-png.flaticon.com/128/281/281760.png',
  '.xls': 'https://cdn-icons-png.flaticon.com/128/281/281765.png',
  '.xlsx': 'https://cdn-icons-png.flaticon.com/128/281/281765.png',
  '.ppt': 'https://cdn-icons-png.flaticon.com/128/888/888074.png',
  '.pptx': 'https://cdn-icons-png.flaticon.com/128/888/888074.png',
  '.ofd': 'https://cdn-icons-png.flaticon.com/128/4823/4823522.png',
  '.wps': 'https://cdn-icons-png.flaticon.com/128/281/281760.png',
}
function fileIcon(ft) { return iconMap[ft] || iconMap['.pdf'] }
function statusClass(s) { return { completed: 'done', pending: 'running', failed: 'failed', deleted: 'failed' }[s] || '' }
function statusText(s) { return { completed: '已解析', pending: '处理中', failed: '解析失败', deleted: '已删除' }[s] || s }

// ════════ Directory Dialog (shell) ════════
const dirVisible = ref(false)
const dirFormRef = ref(null)
const dirForm = reactive({ name: '', parent: '', sort: 0, desc: '' })
const dirRules = {
  name: [{ required: true, message: '请输入目录名称', trigger: 'blur' }],
  parent: [{ required: true, message: '请选择上级目录', trigger: 'change' }],
  sort: [{ required: true, message: '请输入排序号', trigger: 'blur' }],
}
function openDirDialog() { dirForm.name = ''; dirForm.parent = ''; dirForm.sort = 0; dirForm.desc = ''; dirVisible.value = true; setTimeout(() => dirFormRef.value?.clearValidate(), 0) }
function submitDirForm() { dirFormRef.value?.validate(v => { if (!v) return; ElMessage.info('后端暂无目录接口，数据未保存'); dirVisible.value = false }) }

// ════════ Init ════════
onMounted(() => {
  fetchData()
  fetchSummary()
})
</script>

<style scoped>
.kb-layout { display: flex; height: 100%; position: relative; }

/* Sidebar */
.kb-sidebar { width: 220px; min-width: 220px; background: #fff; border-right: 1px solid #ebeef0; padding: 16px 0; }
.sidebar-title { padding: 0 20px 12px; font-size: 15px; font-weight: 600; color: #303133; }
.sidebar-action { display: flex; align-items: center; gap: 6px; padding: 10px 20px; cursor: pointer; color: #0f9c8f; font-size: 13px; transition: background 0.2s; }
.sidebar-action:hover { background: #f2f8f7; }

/* Main */
.kb-main { flex: 1; padding: 16px 24px; overflow-y: auto; }

.stats-bar { display: flex; gap: 20px; padding: 10px 16px; background: #f0f9f8; border-radius: 8px; margin-bottom: 14px; font-size: 14px; }
.stat-warn { color: #e6a23c; }

.filter-row { display: flex; gap: 8px; align-items: center; margin-bottom: 14px; flex-wrap: wrap; }
.search-input { width: 240px; }

/* Table */
.file-name-cell { display: flex; align-items: center; gap: 8px; }
.file-type-icon { width: 20px; height: 20px; flex-shrink: 0; }
.file-name-text { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.doc-number-badge { font-size: 11px; color: #909399; background: #f0f0f0; padding: 1px 6px; border-radius: 3px; white-space: nowrap; }
.catalog-text { font-size: 13px; color: #606266; }

.status-cell { display: flex; align-items: center; gap: 6px; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.status-dot.done { background: #14a89a; }
.status-dot.running { background: #3a8ee6; }
.status-dot.failed { background: #f56c6c; }

.pagination-wrap { display: flex; justify-content: flex-end; margin-top: 16px; }

/* FAB */
.kb-fab { position: fixed; right: 24px; bottom: 40px; width: 60px; height: 60px; border-radius: 50%; background: linear-gradient(135deg, #0f9c8f, #14a89a); box-shadow: 0 4px 14px rgba(15,156,143,.4); display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer; z-index: 200; transition: transform 0.2s, box-shadow 0.2s; }
.kb-fab:hover { transform: scale(1.08); box-shadow: 0 6px 20px rgba(15,156,143,.55); }
.fab-icon { font-size: 22px; }
.fab-label { font-size: 10px; color: #fff; margin-top: 1px; }

/* Upload dialog */
.upload-area { border: 2px dashed #dcdfe6; border-radius: 8px; padding: 40px; text-align: center; cursor: pointer; transition: border-color 0.2s; }
.upload-area:hover { border-color: #0f9c8f; }
.upload-file-list { margin-bottom: 12px; }
.upload-file-item { display: flex; align-items: center; gap: 12px; padding: 8px 12px; background: #fafafa; border-radius: 6px; margin-bottom: 6px; }
.uf-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; }
.uf-status { font-size: 12px; white-space: nowrap; }
.uf-status.pending { color: #909399; }
.uf-status.running { color: #3a8ee6; }
.uf-status.done { color: #14a89a; }
.uf-status.failed { color: #f56c6c; }

.dir-selectors { margin-top: 16px; padding-top: 12px; border-top: 1px solid #ebeef0; }
.dir-label { font-size: 13px; color: #606266; margin-bottom: 8px; display: block; }
.cascade-row { display: flex; gap: 8px; flex-wrap: wrap; }
</style>
