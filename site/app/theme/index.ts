import DefaultTheme from 'vitepress/theme'
import Layout from './Layout.vue'
import HomePage from './components/HomePage.vue'
import ArticleList from './components/ArticleList.vue'
import TopicsPage from './components/TopicsPage.vue'
import MermaidDiagram from './components/MermaidDiagram.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app, router }) {
    app.component('HomePage', HomePage)
    app.component('ArticleList', ArticleList)
    app.component('TopicsPage', TopicsPage)
    app.component('MermaidDiagram', MermaidDiagram)
    const previous = router.onAfterRouteChanged
    router.onAfterRouteChanged = async (href) => {
      await previous?.(href)
      if (typeof window !== 'undefined') {
        window.dispatchEvent(new Event('workshop-route-change'))
      }
    }
  },
}
