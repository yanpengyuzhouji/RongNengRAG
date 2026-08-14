<template>
  <div class="app-layout">
    <!-- Header -->
    <header class="app-header">
      <div class="header-left">
        <span class="logo">⚡ RAG 知识库</span>
        <nav class="header-nav">
          <span
            v-for="item in navItems"
            :key="item.key"
            :class="['nav-item', { active: item.key === activePage && item.page }]"
            @click="handleNav(item)"
          >{{ item.label }}</span>
        </nav>
      </div>
      <div class="header-right">
        <el-icon class="header-icon"><Search /></el-icon>
        <el-badge :value="3" class="header-badge">
          <el-icon class="header-icon"><Bell /></el-icon>
        </el-badge>
        <span class="avatar">👤</span>
        <span class="username">用户</span>
      </div>
    </header>

    <!-- Body -->
    <div class="app-body">
      <!-- Document Preview (overrides all pages) -->
      <DocumentPreview
        v-if="previewFile"
        :file-hash="previewFile.hash"
        :file-name="previewFile.name"
        @close="previewFile = null"
      />
      <KnowledgeBase v-else-if="activePage === 'kb'" @go-ai="activePage = 'ai'" @preview="onPreview" />
      <AIAssistant v-else-if="activePage === 'ai'" />
      <ExcelWorkbench v-else-if="activePage === 'excel'" />
      <div v-else class="empty-page">
        <p>🚧 功能开发中</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import KnowledgeBase from './KnowledgeBase.vue'
import AIAssistant from './AIAssistant.vue'
import DocumentPreview from './DocumentPreview.vue'
import ExcelWorkbench from './excel/ExcelWorkbench.vue'

const activePage = ref('kb')
const previewFile = ref(null)

function onPreview(row) {
  previewFile.value = { hash: row.file_hash, name: row.file_name }
}

const navItems = [
  { key: 'dashboard', label: '数据看板', page: false },
  { key: 'kb', label: '知识库', page: true },
  { key: 'ai', label: 'AI 助手', page: true },
  { key: 'excel', label: 'Excel 分析', page: true },
  { key: 'report', label: '报告生成', page: false },
  { key: 'cost', label: '造价生成', page: false },
  { key: 'review', label: '智能审图', page: false },
  { key: 'rules', label: '规则管理', page: false },
  { key: 'system', label: '系统管理', page: false },
]

function handleNav(item) {
  if (item.key === 'ai' || item.key === 'kb' || item.key === 'excel') {
    activePage.value = item.key
  }
}
</script>

<style>
/* ---- Reset & Global ---- */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
  background: #f5f6f8;
  color: #303133;
}
</style>

<style scoped>
.app-layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

/* ---- Header ---- */
.app-header {
  height: 56px;
  min-height: 56px;
  background: linear-gradient(135deg, #0f9c8f, #14a89a);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  color: #fff;
  z-index: 100;
}
.header-left {
  display: flex;
  align-items: center;
  gap: 32px;
}
.logo {
  font-size: 16px;
  font-weight: 600;
  white-space: nowrap;
}
.header-nav {
  display: flex;
  gap: 8px;
}
.nav-item {
  padding: 4px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background 0.2s;
}
.nav-item:hover { background: rgba(255,255,255,0.15); }
.nav-item.active { background: rgba(255,255,255,0.25); font-weight: 600; }
.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}
.header-icon {
  font-size: 18px;
  cursor: pointer;
  color: #fff;
}
.header-badge { display: flex; align-items: center; }
.avatar {
  width: 28px; height: 28px;
  background: rgba(255,255,255,0.25);
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 14px;
}
.username { font-size: 13px; color: rgba(255,255,255,0.9); }

/* ---- Body ---- */
.app-body {
  flex: 1;
  overflow: hidden;
}

.empty-page {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  font-size: 20px;
  color: #909399;
}
</style>
