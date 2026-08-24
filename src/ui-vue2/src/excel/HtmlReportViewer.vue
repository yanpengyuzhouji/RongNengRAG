<template>
  <div class="html-report-viewer">
    <iframe
      v-if="url"
      ref="reportFrame"
      :src="url"
      class="report-frame"
      loading="lazy"
      sandbox="allow-modals"
      referrerpolicy="no-referrer"
      @load="loaded = true"
    ></iframe>
    <div v-if="url && loaded" class="report-actions">
      <el-button
        type="primary"
        size="small"
        :loading="printing"
        aria-label="下载 PDF"
        @click="downloadPdf"
      >
        <span aria-hidden="true">↓</span>
        下载 PDF
      </el-button>
    </div>
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
const printing = ref(false)
const reportFrame = ref(null)
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

function downloadPdf() {
  const frameWindow = reportFrame.value?.contentWindow
  if (!frameWindow || !loaded.value) return

  printing.value = true
  try {
    // Browsers expose "Save as PDF" from the print dialog.  Printing the
    // iframe keeps the report itself as the only page content and avoids
    // opening an untrusted report in a new top-level window.
    frameWindow.focus()
    frameWindow.print()
  } catch (err) {
    error.value = err?.message || '无法打开 PDF 下载窗口'
  } finally {
    // print() is asynchronous when a dialog is shown; this only controls the
    // short visual loading state of the button.
    window.setTimeout(() => { printing.value = false }, 300)
  }
}
</script>

<style scoped>
.html-report-viewer {
  position: relative;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  overflow: hidden;
  margin: 8px 0;
  background: #fff;
}
.report-frame {
  width: 100%;
  height: 620px;
  border: none;
  display: block;
  background: #fff;
}
.report-actions {
  position: absolute;
  right: 16px;
  bottom: 16px;
  z-index: 2;
  display: flex;
  align-items: center;
  padding: 4px;
  border-radius: 8px;
  background: rgb(255 255 255 / 88%);
  box-shadow: 0 2px 10px rgb(15 23 42 / 16%);
  backdrop-filter: blur(4px);
}
.report-empty {
  padding: 24px 0;
}

@media (max-width: 768px) {
  .report-frame {
    height: 520px;
  }

  .report-actions {
    right: 10px;
    bottom: 10px;
  }
}
</style>
