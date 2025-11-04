export function createElement(type, props, ...children) {
  return {
    type,
    props: {
      ...(props || {}),
      children: children.flat().filter(c => c != null && c !== false).map(c =>
        typeof c === 'object' ? c : createText(c)
      ),
    },
  };
}

function createText(text) {
  return { type: 'TEXT', props: { nodeValue: String(text), children: [] } };
}
