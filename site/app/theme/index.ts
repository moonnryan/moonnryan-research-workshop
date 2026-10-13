import DefaultTheme from 'vitepress/theme'
import Layout from './Layout.vue'
import HomePage from './components/HomePage.vue'
import ArticleList from './components/ArticleList.vue'
import MermaidDiagram from './components/MermaidDiagram.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('HomePage', HomePage)
    app.component('ArticleList', ArticleList)
    app.component('MermaidDiagram', MermaidDiagram)
  },
}
