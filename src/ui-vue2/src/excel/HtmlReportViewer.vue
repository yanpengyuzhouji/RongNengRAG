<template>
  <div class="html-report-viewer">
    <div v-if="title" class="report-bar">
      <span class="report-title">{{ title }}</span>
      <el-button size="small" text @click="openNew">新窗口打开 ↗</el-button>
    </div>
    <iframe
      v-if="url"
      :src="url"
      class="report-frame"
      loading="lazy"
      sandbox
      referrerpolicy="no-referrer"
      @load="loaded = true"
    ></iframe>
    <el-alert v-if="error" :title="error" type="error" :closable="false" show-icon />
    <el-empty v-else-if="!loaded" description="报告加载中…" class="report-empty" />
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { fetchWorkbookReport } from '../api.js'

const props = defineProps({
  workbookId: { type: String, required: true },
  reportId: { type: String, required: true },
  title: { type: String, default: '分析报告' },
})

const loaded = ref(false)
const error = ref('')
const url = ref('')
const controller = new AbortController()

onMounted(async () => {
  try {
    const report = await fetchWorkbookReport(
      props.workbookId, props.reportId, controller.signal
    )
    url.value = URL.createObjectURL(report)
  } catch (err) {
    if (err?.name !== 'AbortError') error.value = err?.message || '报告加载失败'
  }
})

onBeforeUnmount(() => {
  controller.abort()
  if (url.value) URL.revokeObjectURL(url.value)
})

function openNew() {
  if (!url.value) return
  const reportWindow = window.open(url.value, '_blank', 'noopener,noreferrer')
  if (reportWindow) reportWindow.opener = null
}
</script>

<style scoped>
.html-report-viewer {
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  overflow: hidden;
  margin: 8px 0;
}
.report-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 12px;
  background: #f5f7fa;
  border-bottom: 1px solid #e4e7ed;
  font-size: 13px;
}
.report-title {
  font-weight: 600;
  color: #303133;
}
.report-frame {
  width: 100%;
  height: 420px;
  border: none;
  display: block;
}
.report-empty {
  padding: 24px 0;
}
</style>
