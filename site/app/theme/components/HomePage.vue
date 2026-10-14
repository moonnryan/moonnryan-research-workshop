<script setup lang="ts">
import { withBase } from 'vitepress'
import { articles } from '../content'
const recent = articles.slice(0, 3)
</script>

<template>
  <div class="home-page">
    <section class="hero" aria-labelledby="hero-title">
      <div class="hero-left">
        <h1 id="hero-title">Research<br><span class="hero-serif">Workshop</span></h1>
      </div>
      <div class="hero-right">
        <p class="hero-intro">训推、部署、算子与优化<br><span class="hero-more">and more</span></p>
      </div>
    </section>

    <section class="recent-section" aria-labelledby="recent-title">
      <div class="section-heading"><div><span class="section-eyebrow">LATEST NOTES</span><h2 id="recent-title">近期文章</h2></div><a class="subtle-link" :href="withBase('/articles.html')">全部文章 <span aria-hidden="true">↗</span></a></div>
      <div v-if="recent.length" class="recent-list">
        <a v-for="(article, index) in recent" :key="article.url" class="recent-item" :href="withBase(article.url)">
          <span class="article-number">{{ String(index + 1).padStart(2, '0') }}</span>
          <div class="recent-content"><div class="card-meta"><span>{{ article.categoryName }}</span><span class="meta-dot">·</span><span>{{ article.updated }}</span></div><h3>{{ article.title }}</h3><p>{{ article.excerpt }}</p></div>
          <span class="row-arrow" aria-hidden="true">↗</span>
        </a>
      </div>
      <div v-else class="empty-state"><span>文章正在整理中。</span><p>完成的记录会陆续出现在这里。</p></div>
    </section>

    <section class="workshop-note" aria-labelledby="workshop-note-title">
      <div class="workshop-note-heading">
        <h2 id="workshop-note-title">moonnryan 的研究与实践</h2>
        <a class="subtle-link" :href="withBase('/about.html')">关于这个 Workshop ↗</a>
      </div>
      <div class="workshop-note-description">
        <p>理解原理，验证方法</p>
        <p>记录技术问题的分析过程、实现取舍与适用边界</p>
      </div>
    </section>
  </div>
</template>
