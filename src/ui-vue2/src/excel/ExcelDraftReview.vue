<template>
  <div class="excel-review">
    <el-card shadow="never" class="review-card">
      <template #header>
        <div class="review-header">
          <span>🔍 审核草稿 — {{ importData?.file_name }}</span>
          <div>
            <el-tag size="small" :type="statusTagType">{{ statusLabel }}</el-tag>
            <el-tag size="small" type="info" class="rev-tag">revision {{ importData?.revision }}</el-tag>
          </div>
        </div>
      </template>

      <!-- Errors -->
      <el-alert
        v-if="errorText"
        type="error"
        :closable="false"
        :title="errorText"
        class="block"
      />

      <!-- Sheet list -->
      <div class="sheet-list">
        <div
          v-for="sheet in importData?.sheets || []"
          :key="sheet.sheet_id"
          :class="['sheet-item', { active: activeSheetId === sheet.sheet_id }]"
          @click="selectSheet(sheet)"
        >
          <span class="sheet-kind" :class="sheet.kind">{{ kindLabel(sheet.kind) }}</span>
          <span class="sheet-name">{{ sheet.name }}</span>
          <span class="sheet-count">{{ sheet.row_count }}行</span>
        </div>
      </div>

      <!-- Active sheet editor -->
      <template v-if="activeSheet">
        <el-divider content-position="left">
          工作表: {{ activeSheet.name }}
          <span class="divider-hint">{{ activeSheet.kind === 'data' ? '表格' : '元数据/忽略' }}</span>
        </el-divider>

        <el-form label-width="90px" size="small" class="sheet-form">
          <el-form-item label="启用">
            <el-switch v-model="activeSheet.enabled" @change="pushSheetUpdate" />
          </el-form-item>
          <el-form-item label="类型">
            <el-select v-model="activeSheet.kind" style="width:160px" @change="pushSheetUpdate">
              <el-option label="数据表" value="data" />
              <el-option label="元数据" value="metadata" />
              <el-option label="忽略" value="ignored" />
            </el-select>
          </el-form-item>
          <el-form-item v-if="activeSheet.kind === 'data'" label="表名">
            <el-input v-model="activeSheet.table_name" style="width:240px" @blur="pushSheetUpdate" />
          </el-form-item>
          <el-form-item v-if="activeSheet.kind === 'data'" label="表头行">
            <el-input-number v-model="activeSheet.header_row" :min="1" @change="pushSheetUpdate" />
          </el-form-item>
          <el-form-item v-if="activeSheet.kind === 'data'" label="数据起始行">
            <el-input-number v-model="activeSheet.data_start_row" :min="1" @change="pushSheetUpdate" />
          </el-form-item>
        </el-form>

        <!-- Columns table (data sheets) -->
        <template v-if="activeSheet.kind === 'data' && activeSheet.columns?.length">
          <div class="cols-title">字段</div>
          <el-table :data="activeSheet.columns" size="small" border>
            <el-table-column label="启用" width="60" align="center">
              <template #default="{ row }">
                <el-switch v-model="row.enabled" size="small" @change="pushColumnUpdate(row)" />
              </template>
            </el-table-column>
            <el-table-column label="显示名" min-width="120">
              <template #default="{ row }">
                <el-input v-model="row.display_name" size="small" @blur="pushColumnUpdate(row)" />
              </template>
            </el-table-column>
            <el-table-column label="物理名" min-width="120">
              <template #default="{ row }">
                <el-input v-model="row.physical_name" size="small" @blur="pushColumnUpdate(row)" />
              </template>
            </el-table-column>
            <el-table-column label="类型" width="110">
              <template #default="{ row }">
                <el-select v-model="row.sql_type" size="small" @change="pushColumnUpdate(row)">
                  <el-option label="TEXT" value="TEXT" />
                  <el-option label="INTEGER" value="INTEGER" />
                  <el-option label="REAL" value="REAL" />
                </el-select>
              </template>
            </el-table-column>
            <el-table-column label="空表头" width="70" align="center">
              <template #default="{ row }">
                <el-tag v-if="row.empty_header" type="warning" size="small">空</el-tag>
                <span v-else>—</span>
              </template>
            </el-table-column>
            <el-table-column label="混合数字" width="80" align="center">
              <template #default="{ row }">
                <el-tag v-if="row.mixed_numeric" type="warning" size="small">混合</el-tag>
                <span v-else>—</span>
              </template>
            </el-table-column>
          </el-table>
        </template>

        <!-- Data preview -->
        <div v-if="activeSheet.kind === 'data'" class="cols-title">
          数据预览
          <el-button link size="small" @click="loadPreview">刷新</el-button>
        </div>
        <el-table
          v-if="previewRows.length"
          :data="previewRows"
          size="small"
          border
          max-height="260"
        >
          <el-table-column v-for="c in previewColumns" :key="c" :prop="c" :label="c" :min-width="90" />
        </el-table>
        <el-empty v-else-if="activeSheet.kind === 'data' && previewLoaded" description="无预览数据" :image-size="60" />
      </template>

      <!-- Warnings -->
      <template v-if="warnings.length">
        <el-divider content-position="left">需要确认的警告 ({{ acceptedWarningCount }}/{{ warningCodes.length }})</el-divider>
        <el-checkbox :model-value="allWarningsAccepted" @change="toggleAllWarnings">全部接受</el-checkbox>
        <div v-for="(w, index) in warnings" :key="`${w.code}-${w.sheet_id || ''}-${w.row || ''}-${index}`" class="warn-item">
          <el-checkbox :model-value="acceptedCodes.includes(w.code)" @change="(v) => toggleWarning(w.code, v)">
            <span class="warn-code">{{ w.code }}</span>
            {{ w.message }}
          </el-checkbox>
        </div>
      </template>

      <template #footer>
        <div class="review-footer">
          <el-button @click="emit('cancel')">返回</el-button>
          <el-button @click="revalidate" :loading="saving">重新校验</el-button>
          <el-button type="primary" :loading="saving" :disabled="hasErrors || !allWarningsAccepted" @click="confirm">
            确认建库
          </el-button>
        </div>
      </template>
    </el-card>
  </div>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue'
import { confirmImport, getSheetRows, getWorkbookImport, updateDraft, validateImport } from '../api.js'
import { acceptedWarningCodeCount, areAllWarningCodesAccepted, uniqueWarningCodes } from './warningAcceptance.js'

const props = defineProps({
  importData: { type: Object, required: true },
})
const emit = defineEmits(['cancel', 'activated'])

const local = reactive({ ...props.importData })
watch(() => props.importData, (v) => { Object.assign(local, v) })

const activeSheetId = ref(local.sheets?.[0]?.sheet_id)
const previewRows = ref([])
const previewColumns = ref([])
const previewLoaded = ref(false)
const saving = ref(false)
const errorText = ref('')
const acceptedCodes = ref([])

const activeSheet = computed(() => local.sheets?.find((s) => s.sheet_id === activeSheetId.value))
const warnings = computed(() => local.validation?.warnings || [])
const warningCodes = computed(() => uniqueWarningCodes(warnings.value))
const errors = computed(() => local.validation?.errors || [])
const hasErrors = computed(() => (local.validation?.error_count || 0) > 0)
const acceptedWarningCount = computed(() => acceptedWarningCodeCount(warnings.value, acceptedCodes.value))
const allWarningsAccepted = computed(() => areAllWarningCodesAccepted(warnings.value, acceptedCodes.value))

const statusLabel = computed(() => ({
  ready: '可建库', draft: '需修复', failed: '失败', uploading: '解析中', active: '已激活',
})[local.status] || local.status)
const statusTagType = computed(() => ({ ready: 'success', draft: 'warning', failed: 'danger', active: 'success' })[local.status] || 'info')

function kindLabel(k) {
  return { data: '表', metadata: '元', ignored: '忽略' }[k] || k
}

function selectSheet(sheet) {
  activeSheetId.value = sheet.sheet_id
  loadPreview()
}

function sheetUpdates() {
  const s = activeSheet.value
  if (!s) return []
  return [{
    sheet_id: s.sheet_id,
    enabled: s.enabled,
    kind: s.kind,
    ...(s.kind === 'data' ? {
      table_name: s.table_name,
      header_row: s.header_row,
      data_start_row: s.data_start_row,
    } : {}),
  }]
}

async function pushSheetUpdate() {
  await save({ sheet_updates: sheetUpdates() })
}

async function pushColumnUpdate(col) {
  await save({ column_updates: [{ sheet_id: activeSheet.value.sheet_id, column_id: col.column_id, enabled: col.enabled, display_name: col.display_name, physical_name: col.physical_name, sql_type: col.sql_type }] })
}

async function save(payload) {
  saving.value = true
  errorText.value = ''
  try {
    const data = await updateDraft(local.import_id, { revision: local.revision, ...payload })
    Object.assign(local, data)
  } catch (e) {
    errorText.value = '保存失败(可能草稿已被他人修改): ' + (e?.message || e)
    await reloadLatest()
  } finally {
    saving.value = false
  }
}

async function reloadLatest() {
  const data = await getWorkbookImport(local.import_id)
  Object.assign(local, data)
}

async function loadPreview() {
  const s = activeSheet.value
  if (!s || s.kind !== 'data') return
  const data = await getSheetRows(local.import_id, s.sheet_id, 0, 50)
  previewRows.value = (data?.rows || []).map((r) => {
    const obj = {}
    ;(data.columns || []).forEach((c, i) => { obj[c.physical_name || c.column_id] = r.values?.[i] ?? '' })
    return obj
  })
  previewColumns.value = (data.columns || []).map((c) => c.physical_name || c.column_id)
  previewLoaded.value = true
}

function toggleWarning(code, val) {
  if (val && !acceptedCodes.value.includes(code)) acceptedCodes.value.push(code)
  if (!val) acceptedCodes.value = acceptedCodes.value.filter((c) => c !== code)
}

function toggleAllWarnings(val) {
  acceptedCodes.value = val ? [...warningCodes.value] : []
}

async function revalidate() {
  saving.value = true
  try {
    const data = await validateImport(local.import_id)
    Object.assign(local, data)
  } catch (e) {
    errorText.value = '校验失败: ' + (e?.message || e)
  } finally {
    saving.value = false
  }
}

async function confirm() {
  saving.value = true
  errorText.value = ''
  try {
    const data = await confirmImport(local.import_id, {
      revision: local.revision,
      accepted_warning_codes: acceptedCodes.value,
    })
    emit('activated', { ...data, file_name: local.file_name, display_name: local.file_name })
  } catch (e) {
    errorText.value = '建库失败: ' + (e?.message || e)
    await reloadLatest()
  } finally {
    saving.value = false
  }
}
</script>

<style scoped>
.excel-review { height: 100%; }
.review-card { border-radius: 8px; }
.review-header { display: flex; align-items: center; justify-content: space-between; }
.rev-tag { margin-left: 8px; }
.block { margin-bottom: 12px; }
.sheet-list { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 4px; }
.sheet-item {
  padding: 6px 12px; border: 1px solid #e4e7ed; border-radius: 6px;
  cursor: pointer; font-size: 13px; display: flex; gap: 8px; align-items: center;
}
.sheet-item.active { border-color: #0f9c8f; background: #e8f7f5; }
.sheet-kind { font-size: 11px; padding: 1px 6px; border-radius: 3px; }
.sheet-kind.data { background: #e1f3d8; color: #67c23a; }
.sheet-kind.metadata { background: #fdf6ec; color: #e6a23c; }
.sheet-kind.ignored { background: #f4f4f5; color: #909399; }
.sheet-count { color: #909399; font-size: 12px; }
.divider-hint { font-size: 12px; color: #909399; }
.sheet-form { max-width: 420px; }
.cols-title { font-weight: 600; font-size: 13px; margin: 10px 0 6px; }
.warn-item { padding: 4px 0; font-size: 13px; }
.warn-code { color: #e6a23c; margin-right: 6px; font-weight: 600; }
.review-footer { text-align: right; }
</style>
