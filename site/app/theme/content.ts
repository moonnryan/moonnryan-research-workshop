import data from '../../data/articles.json'

export const articles = data.articles
export const isPreview = data.preview
export const categories = [...new Map(articles.map(article => [article.category, article.categoryName])).entries()]
