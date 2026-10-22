export const topicLabels = {
  'quantization-accuracy': '量化与精度',
  'graphs-operators': '计算图与算子',
  tutorial: '操作教程',
  'engineering-runtime': '工程与运行环境',
  'training-deployment': '训练与部署',
}

const aliases = {
  'models-training': 'training-deployment',
  'inference-performance': 'training-deployment',
  'data-evaluation': 'training-deployment',
}

export function normalizeTopic(topic) {
  return aliases[topic] ?? topic
}
