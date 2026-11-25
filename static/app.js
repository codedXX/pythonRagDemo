// 问答页面通过本应用的接口收发数据，服务端负责与模型供应商通信。

const els = {  // 集中保存页面控件的引用。
  messages: document.getElementById('messages'),  // 定位对话消息区。
  composer: document.getElementById('composer'),  // 定位提问表单。
  question: document.getElementById('question'),  // 取得问题输入框。
  send: document.getElementById('send'),  // 取得发送按钮。
  dropzone: document.getElementById('dropzone'),  // 取得文件拖放区。
  fileInput: document.getElementById('file-input'),  // 取得文件选择器。
  uploadList: document.getElementById('upload-list'),  // 取得上传进度列表。
  docList: document.getElementById('doc-list'),  // 取得知识库文档列表。
  refreshDocs: document.getElementById('refresh-docs'),  // 取得刷新文档按钮。
  healthPanel: document.getElementById('health-panel'),  // 取得依赖状态面板。
  healthToggle: document.getElementById('health-toggle'),  // 取得面板开关。
  debugCheck: document.getElementById('debug-check'),  // 取得调试复选框。
};  // 结束定义。

// 由客户端保存对话记录，服务端可以无会话状态运行。
const history = [];  // 在浏览器内保存最近的问答。
let busy = false;  // 初始时允许提问。

// 页面元素与标记文本渲染工具。

function el(tag, className, text) {  // 按标签、样式和文字创建元素。
  const node = document.createElement(tag);  // 创建指定标签的节点。
  if (className) node.className = className;  // 有样式类时应用到节点。
  if (text !== undefined) node.textContent = text;  // 用纯文本填充内容。
  return node;  // 返回创建的元素。
}  // 结束当前代码块。

function escapeHtml(text) {  // 转义可能被浏览器解释为标记的字符。
  return text  // 以输入文本为起点依次替换。
    .replace(/&/g, '&amp;')  // 先转义与号，避免后续实体被重复编码。
    .replace(/</g, '&lt;')  // 转义左尖括号。
    .replace(/>/g, '&gt;')  // 转义右尖括号。
    .replace(/"/g, '&quot;');  // 转义双引号。
}  // 结束当前代码块。

// 先转义来自文档和模型的文字，再为支持的语法生成页面标签。
function renderMarkdown(source) {  // 把受支持的标记语法渲染为回答正文。
  const blocks = [];  // 暂存已转义的完整代码块。
  // 先保护完整代码块，避免后续的行内样式替换改变代码内容。
  let text = source.replace(/```([\w+-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {  // 提取围栏中的语言标记和代码。
    blocks.push(  // 保存独立渲染的代码块。
      `<pre><code>${escapeHtml(code.replace(/\n$/, ''))}</code></pre>`  // 去掉代码末尾的换行并转义后包入预格式化标签。
    );  // 结束当前调用。
    return `\u0000BLOCK${blocks.length - 1}\u0000`;  // 用带序号的占位符保留代码块位置。
  });  // 结束本次调用。

  const lines = text.split('\n');  // 逐行拆分剩余正文。
  const out = [];  // 按顺序收集渲染结果。
  let listType = null;  // 初始时没有打开的列表。
  let table = null;  // 初始时没有待输出的表格。

  const closeList = () => {  // 收尾当前列表。
    if (listType) {  // 仅在列表已打开时处理。
      out.push(`</${listType}>`);  // 追加对应类型的闭合标签。
      listType = null;  // 清除当前列表类型。
    }  // 结束当前代码块。
  };  // 结束定义。
  const closeTable = () => {  // 输出缓存中的表格行。
    if (table) {  // 检查是否有待输出的表格。
      out.push('<table>');  // 写入表格开始标签。
      table.forEach((row, index) => {  // 依次渲染每行并取得行号。
        const tag = index === 0 ? 'th' : 'td';  // 首行使用表头，其余行使用普通单元格。
        out.push(`<tr>${row.map((cell) => `<${tag}>${inline(cell)}</${tag}>`).join('')}</tr>`);  // 逐格渲染行内标记并拼成一行。
      });  // 结束本次调用。
      out.push('</table>');  // 闭合表格。
      table = null;  // 清空表格状态，准备收集下一张表。
    }  // 结束当前代码块。
  };  // 结束定义。

  function inline(value) {  // 处理单行中的样式、引用和链接。
    return escapeHtml(value)  // 先转义文本，再生成受支持的标签。
      .replace(/`([^`]+)`/g, '<code>$1</code>')  // 识别反引号包裹的行内代码。
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')  // 识别双星号包裹的加粗文字。
      .replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>')  // 识别单星号包裹的斜体文字。
      .replace(/\[(\d+)\]/g, '<span class="citation">[$1]</span>')  // 将方括号数字渲染为引用编号。
      .replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');  // 将网页链接渲染为在新标签页打开的锚点。
  }  // 结束当前代码块。

  for (const raw of lines) {  // 按原顺序遍历正文各行。
    const line = raw;  // 保留当前行原文用于结构识别。
    if (/^\u0000BLOCK\d+\u0000$/.test(line.trim())) {  // 识别先前保存的独立代码块占位符。
      closeList();  // 闭合前一个列表。
      closeTable();  // 输出前面累积的表格。
      out.push(line.trim());  // 把代码块占位符直接放入结果。
      continue;  // 当前项已处理，继续下一项。
    }  // 结束当前代码块。
    if (/^\s*\|.*\|\s*$/.test(line)) {  // 识别两端由竖线包围的表格行。
      const cells = line.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());  // 去掉外侧竖线并拆分、清理单元格。
      // 表格的分隔行标记对齐方式，本身无需显示。
      if (!cells.every((c) => /^:?-{2,}:?$/.test(c))) {  // 跳过仅由连字符和对齐标记组成的分隔行。
        closeList();  // 闭合前一个列表。
        table = table || [];  // 首次遇到表格时初始化行缓存。
        table.push(cells);  // 保存当前行的单元格。
        continue;  // 当前项已处理，继续下一项。
      }  // 结束当前代码块。
      continue;  // 当前项已处理，继续下一项。
    }  // 结束当前代码块。
    closeTable();  // 输出前面累积的表格。

    if (!line.trim()) {  // 把空行作为列表边界。
      closeList();  // 闭合前一个列表。
      continue;  // 当前项已处理，继续下一项。
    }  // 结束当前代码块。
    const heading = line.match(/^(#{1,6})\s+(.*)$/);  // 提取标题级别与标题正文。
    if (heading) {  // 单独渲染标题行。
      closeList();  // 闭合前一个列表。
      const level = Math.min(3, heading[1].length);  // 页面标题最多显示为三级。
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);  // 使用对应级别的标签输出标题正文。
      continue;  // 当前项已处理，继续下一项。
    }  // 结束当前代码块。
    if (/^\s*>\s?/.test(line)) {  // 识别带引用前缀的行。
      closeList();  // 闭合前一个列表。
      out.push(`<blockquote>${inline(line.replace(/^\s*>\s?/, ''))}</blockquote>`);  // 去掉引用前缀并渲染引文。
      continue;  // 当前项已处理，继续下一项。
    }  // 结束当前代码块。
    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);  // 识别无序列表项。
    const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);  // 识别带数字的有序列表项。
    if (bullet || ordered) {  // 处理两种列表项。
      const wanted = bullet ? 'ul' : 'ol';  // 根据前缀选用有序或无序列表。
      if (listType !== wanted) {  // 列表类型改变时先结束旧列表。
        closeList();  // 闭合前一个列表。
        out.push(`<${wanted}>`);  // 写入新列表的开始标签。
        listType = wanted;  // 记住所处的列表类型。
      }  // 结束当前代码块。
      out.push(`<li>${inline((bullet || ordered)[1])}</li>`);  // 提取列表项正文并处理行内样式。
      continue;  // 当前项已处理，继续下一项。
    }  // 结束当前代码块。
    closeList();  // 闭合前一个列表。
    out.push(`<p>${inline(line)}</p>`);  // 普通行作为一个段落输出。
  }  // 结束当前代码块。
  closeList();  // 闭合前一个列表。
  closeTable();  // 输出前面累积的表格。

  let html = out.join('\n');  // 拼接已渲染的各行。
  html = html.replace(/\u0000BLOCK(\d+)\u0000/g, (_, index) => blocks[Number(index)]);  // 把占位符换回原位置的代码块。
  return html;  // 返回渲染结果。
}  // 结束当前代码块。

// 外部依赖状态。

function statusClass(status) {  // 为依赖状态选择显示样式。
  if (status === 'healthy') return 'health-ok';  // 正常时应用成功样式。
  if (status === 'misconfigured') return 'health-warn';  // 配置异常时应用警告样式。
  return 'health-bad';  // 其他异常应用故障样式。
}  // 结束当前代码块。

async function loadHealth() {  // 向服务端读取最新依赖状态。
  try {  // 尝试执行本次操作。
    const response = await fetch('/api/health');  // 请求健康检查接口。
    const data = await response.json();  // 解析响应中的状态数据。
    els.healthPanel.replaceChildren();  // 清除上一次检查的显示结果。

    for (const dep of data.dependencies || []) {  // 逐个处理服务端返回的依赖项。
      const row = el('div', 'health-row');  // 创建一行状态信息。
      row.append(el('span', 'health-name', dep.name.replace(/_/g, ' ')));  // 将名称中的下划线替换为空格后展示。
      row.append(el('span', statusClass(dep.status), dep.status));  // 按状态显示文字和颜色。
      els.healthPanel.append(row);  // 将状态行加入面板。
      if (dep.status !== 'healthy' && dep.message) {  // 仅在异常且有说明时显示详情。
        els.healthPanel.append(el('div', 'health-msg', dep.message));  // 追加服务端给出的故障原因。
      }  // 结束当前代码块。
    }  // 结束当前代码块。
    if (data.dimension) {  // 有实测向量维度时额外显示。
      const row = el('div', 'health-row');  // 创建一行状态信息。
      row.append(el('span', 'health-name', 'vector dim'));  // 添加向量维度标签。
      row.append(el('span', 'health-ok', String(data.dimension)));  // 显示实测维度值。
      els.healthPanel.append(row);  // 将状态行加入面板。
    }  // 结束当前代码块。
  } catch (error) {  // 处理操作过程中的异常。
    els.healthPanel.replaceChildren(el('div', 'health-msg', `无法获取状态：${error.message}`));  // 用失败原因替换状态面板内容。
  }  // 结束当前代码块。
}  // 结束当前代码块。

els.healthToggle.addEventListener('click', async () => {  // 点击状态按钮时切换面板。
  const hidden = els.healthPanel.classList.toggle('hidden');  // 切换隐藏样式并取得切换后的状态。
  if (!hidden) await loadHealth();  // 展开时重新读取状态。
});  // 结束本次调用。

// 用户开始操作前，先提示已知的依赖配置问题。
async function warnIfUnhealthy() {  // 在首次操作前提示依赖配置问题。
  try {  // 尝试执行本次操作。
    const data = await (await fetch('/api/health')).json();  // 读取服务端健康检查结果。
    const broken = (data.dependencies || []).filter((d) => d.status === 'misconfigured');  // 筛出配置异常的依赖。
    if (broken.length === 0) return false;  // 没有异常配置时直接结束检查。
    els.healthPanel.classList.remove('hidden');  // 展开状态面板。
    await loadHealth();  // 刷新依赖状态。
    return true;  // 告知调用方已展示提示。
  } catch {  // 处理无需取用异常对象的失败。
    return false;  // 告知调用方未展示提示。
  }  // 结束当前代码块。
}  // 结束当前代码块。

// 知识库文档列表。

function describeDoc(doc) {  // 生成文档状态摘要。
  if (doc.state === 'failed') return doc.error || '处理失败';  // 失败时优先展示具体原因。
  if (doc.state === 'staging') return '处理中…';  // 入库过程中显示等待提示。
  return `${doc.parent_count} 段 / ${doc.child_count} 块`;  // 完成后显示父块和子块数量。
}  // 结束当前代码块。

async function loadDocuments() {  // 刷新知识库文档列表。
  try {  // 尝试执行本次操作。
    const response = await fetch('/api/docs');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const { documents } = await response.json();
    if (!Array.isArray(documents)) throw new Error('响应缺少文档列表');
    els.docList.replaceChildren();
    if (documents.length === 0) {
      els.docList.append(el('li', 'doc-sub', '还没有文档'));  // 显示空列表提示。
      return;  // 结束当前处理。
    }  // 结束当前代码块。
    for (const doc of documents) {  // 依次展示各份文档。
      const item = el('li', `doc-item ${doc.state}`);  // 按文档状态创建列表项。
      const meta = el('div', 'doc-meta');  // 创建文档摘要容器。
      meta.append(el('span', 'doc-name', doc.name));  // 显示文件名。
      meta.append(el('span', 'doc-sub', describeDoc(doc)));  // 展示状态或切块统计。
      item.append(meta);  // 将摘要加到文档项中。

      const remove = el('button', 'doc-remove', '×');  // 创建删除按钮。
      remove.title = '删除';  // 设置删除按钮的悬停提示。
      remove.disabled = doc.state === 'staging';  // 入库期间禁用删除。
      remove.addEventListener('click', () => removeDocument(doc, remove));  // 点击时删除此文档。
      item.append(remove);  // 将删除按钮加入文档项。
      els.docList.append(item);  // 将文档项加入页面列表。
    }  // 结束当前代码块。
  } catch (error) {  // 处理操作过程中的异常。
    els.docList.replaceChildren(el('li', 'doc-sub', `加载失败：${error.message}`));  // 在列表位置显示加载失败原因。
  }  // 结束当前代码块。
}  // 结束当前代码块。

async function removeDocument(doc, button) {  // 向服务端请求删除指定文档。
  button.disabled = true;  // 暂时禁用按钮以避免重复提交。
  try {  // 尝试执行本次操作。
    const response = await fetch(`/api/docs/${encodeURIComponent(doc.doc_id)}`, {  // 用编码后的文档标识构造删除地址。
      method: 'DELETE',  // 使用删除请求方法。
    });  // 结束本次调用。
    if (!response.ok) throw new Error(`HTTP ${response.status}`);  // 将失败响应转为带状态码的异常。
    await loadDocuments();  // 重新读取知识库列表。
    await loadHealth();  // 刷新依赖状态。
  } catch (error) {  // 处理操作过程中的异常。
    button.disabled = false;  // 失败后允许再次删除。
    button.title = `删除失败：${error.message}`;  // 将错误原因显示在按钮提示中。
  }  // 结束当前代码块。
}  // 结束当前代码块。

els.refreshDocs.addEventListener('click', loadDocuments);  // 点击刷新按钮时重新载入文档。

// 文件上传与进度反馈。

function uploadRow(name) {  // 为单个待上传文件创建进度行。
  const item = el('li', 'upload-item');  // 创建上传列表项。
  item.dataset.filename = name;  // 保存此行关联的文件名。

  const head = el('div', 'upload-item-head');  // 创建上传项标题行。
  head.append(el('span', 'upload-name', name));  // 在标题中显示文件名。
  head.append(el('span', 'upload-state', '等待中'));  // 显示初始等待状态。
  item.append(head);  // 将标题行加入当前项目。

  const track = el('div', 'progress-track');  // 创建进度条轨道。
  const bar = el('div', 'progress-bar');  // 创建表示完成比例的色条。
  track.append(bar);  // 将色条放进轨道。
  item.append(track);  // 将进度条加入上传项。

  const detail = el('p', 'upload-detail', '');  // 创建进度详情文本区域。
  item.append(detail);  // 将详情区域加入上传项。

  els.uploadList.append(item);  // 在页面上显示此上传项。
  return { item, state: head.lastElementChild, bar, detail };  // 返回后续更新进度所需的元素引用。
}  // 结束当前代码块。

const STAGE_LABEL = {  // 把后端阶段值映射为中文提示。
  parsing: '解析中',  // 显示文件解析阶段。
  chunking: '分块中',  // 显示文档切块阶段。
  embedding: '向量化中',  // 显示向量生成阶段。
  persisting: '入库中',  // 显示存储写入阶段。
  done: '完成',  // 显示成功状态。
  failed: '失败',  // 显示失败状态。
};  // 结束定义。

async function uploadFiles(files) {  // 校验文件并批量上传，随后读取进度流。
  if (!files || files.length === 0) return;  // 未选择文件时直接结束。

  const allowed = [];  // 收集格式受支持的文件。
  for (const file of files) {  // 逐个检查所选文件。
    const ext = (file.name.split('.').pop() || '').toLowerCase();  // 提取扩展名并转成小写。
    if (!['pdf', 'md', 'markdown'].includes(ext)) {  // 识别接口不支持的扩展名。
      const row = uploadRow(file.name);  // 为此文件建立反馈行。
      row.item.classList.add('failed');  // 应用失败样式。
      row.state.textContent = '不支持';  // 标明格式不受支持。
      row.detail.textContent = `不支持 .${ext}；请上传 PDF 或 Markdown（.md/.markdown）`;  // 提示可接受的文档格式。
      continue;  // 当前项已处理，继续下一项。
    }  // 结束当前代码块。
    allowed.push(file);  // 将通过检查的文件加入上传批次。
  }  // 结束当前代码块。
  if (allowed.length === 0) return;  // 没有可上传的文件时结束。

  const form = new FormData();  // 创建多文件表单。
  for (const file of allowed) form.append('files', file);  // 将每个文件追加到同一表单字段。

  const rows = new Map();  // 建立文件名与进度行的对应关系。
  for (const file of allowed) rows.set(file.name, uploadRow(file.name));  // 为每个受支持的文件创建进度行。

  try {  // 尝试执行本次操作。
    const response = await fetch('/api/docs', { method: 'POST', body: form });  // 提交表单并等待上传事件流。
    if (!response.ok || !response.body) {  // 检查响应是否成功且包含可读数据流。
      throw new Error(`HTTP ${response.status}`);  // 抛出包含响应状态码的异常。
    }  // 结束当前代码块。

    const reader = response.body.getReader();  // 取得响应数据流的读取器。
    const decoder = new TextDecoder();  // 创建支持跨数据块解码的文本解码器。
    let buffer = '';  // 缓存尚未拼成完整事件的数据。

    while (true) {  // 持续读取直至响应流结束。
      const { done, value } = await reader.read();  // 取出下一批字节及结束标记。
      if (done) break;  // 读到流末尾就退出。
      buffer += decoder.decode(value, { stream: true });  // 保留解码状态并把新文本接到缓冲区。

      // 每个完整推送事件以一个空行结束。
      let boundary = buffer.indexOf('\n\n');  // 查找标记事件结尾的空行。
      while (boundary !== -1) {  // 缓冲区含完整事件时逐帧处理。
        const frame = buffer.slice(0, boundary);  // 取出分隔符之前的完整事件。
        buffer = buffer.slice(boundary + 2);  // 从缓冲区移除已处理的事件及分隔符。
        handleUploadEvent(frame, rows);  // 将完整上传事件交给对应的进度行。
        boundary = buffer.indexOf('\n\n');  // 继续查找下一个完整事件。
      }  // 结束当前代码块。
    }  // 结束当前代码块。
  } catch (error) {  // 处理操作过程中的异常。
    for (const [name, row] of rows) {  // 逐个检查当前上传批次的进度行。
      if (!row.item.classList.contains('done')) {  // 仅将尚未成功的文件标成失败。
        row.item.classList.add('failed');  // 应用失败样式。
        row.state.textContent = '失败';  // 更新状态文字。
        row.detail.textContent = `请求失败：${error.message}`;  // 在进度行展示请求失败的原因。
      }  // 结束当前代码块。
    }  // 结束当前代码块。
  } finally {  // 无论请求成功或失败都执行收尾操作。
    await loadDocuments();  // 重新读取知识库列表。
    await loadHealth();  // 刷新依赖状态。
  }  // 结束当前代码块。
}  // 结束当前代码块。

function parseFrame(frame) {  // 拆解服务端推送的单个事件。
  let event = 'message';  // 未声明事件名时使用默认名称。
  const dataLines = [];  // 收集多行事件载荷。
  for (const line of frame.split('\n')) {  // 逐行读取事件帧。
    if (line.startsWith('event: ')) event = line.slice(7).trim();  // 从事件名字段中提取名称。
    else if (line.startsWith('data: ')) dataLines.push(line.slice(6));  // 去掉载荷前缀，保留后续正文。
  }  // 结束当前代码块。
  if (dataLines.length === 0) return null;  // 忽略没有有效载荷的帧。
  try {  // 尝试执行本次操作。
    return { event, data: JSON.parse(dataLines.join('\n')) };  // 拼接载荷并解析其结构化数据。
  } catch {  // 处理无需取用异常对象的失败。
    return null;  // 忽略无法解析的事件。
  }  // 结束当前代码块。
}  // 结束当前代码块。

function handleUploadEvent(frame, rows) {  // 根据上传事件更新对应文件的界面。
  const parsed = parseFrame(frame);  // 解析当前帧。
  if (!parsed) return;  // 忽略无有效数据的帧。
  const { event, data } = parsed;  // 取出事件名和载荷。
  const row = rows.get(data.name);  // 用文件名找到对应进度行。
  if (!row) return;  // 忽略不属于当前批次的事件。

  if (event === 'progress') {  // 处理进度通知。
    const label = STAGE_LABEL[data.stage] || data.stage;  // 优先显示中文阶段名，未知阶段使用原值。
    row.state.textContent = label;  // 更新当前处理阶段。
    if (typeof data.fraction === 'number') {  // 存在数值进度时更新进度条。
      row.bar.style.width = `${Math.round(data.fraction * 100)}%`;  // 将完成比例转成整数百分比。
    }  // 结束当前代码块。
    row.detail.textContent = data.message || '';  // 显示服务端给出的阶段说明。
    if (data.stage === 'embedding' && data.total) {  // 向量阶段有总数时按实际处理条数计算。
      row.bar.style.width = `${Math.round((data.current / data.total) * 100)}%`;  // 根据已完成和总条数计算进度条宽度。
    }  // 结束当前代码块。
  } else if (event === 'done') {  // 处理正常结束通知。
    row.item.classList.remove('failed');  // 清除旧的失败样式。
    row.item.classList.add('done');  // 应用完成样式。
    row.bar.style.width = '100%';  // 填满进度条。
    row.state.textContent = '完成';  // 显示完成状态。
    row.detail.textContent = `${data.parents} 段 / ${data.children} 块，用时 ${data.duration}s`;  // 显示切分统计和耗时。
  } else if (event === 'failed') {  // 处理入库失败通知。
    row.item.classList.add('failed');  // 应用失败样式。
    row.state.textContent = '失败';  // 更新状态文字。
    row.detail.textContent = `${STAGE_LABEL[data.stage] || data.stage}：${data.error}`;  // 同时显示失败阶段与原因。
  } else if (event === 'rejected') {  // 处理上传被拒绝的通知。
    row.item.classList.add('failed');  // 应用失败样式。
    row.state.textContent = '拒绝';  // 显示拒绝状态。
    row.detail.textContent = data.error;  // 显示服务端给出的拒绝原因。
  }  // 结束当前代码块。
}  // 结束当前代码块。

els.dropzone.addEventListener('click', () => els.fileInput.click());  // 点击拖放区时打开文件选择器。
els.dropzone.addEventListener('keydown', (event) => {  // 允许用键盘操作拖放区。
  if (event.key === 'Enter' || event.key === ' ') {  // 响应回车键或空格键。
    event.preventDefault();  // 阻止浏览器执行默认操作。
    els.fileInput.click();  // 打开系统文件选择器。
  }  // 结束当前代码块。
});  // 结束本次调用。
els.fileInput.addEventListener('change', () => {  // 选择文件后开始处理。
  uploadFiles(els.fileInput.files);  // 上传选中的文件。
  els.fileInput.value = '';  // 清空选择值，方便再次选择同一文件。
});  // 结束本次调用。

for (const type of ['dragenter', 'dragover']) {  // 监听文件进入或停留在拖放区。
  els.dropzone.addEventListener(type, (event) => {  // 注册对应的拖放事件处理函数。
    event.preventDefault();  // 阻止浏览器执行默认操作。
    els.dropzone.classList.add('dragover');  // 高亮当前拖放目标。
  });  // 结束本次调用。
}  // 结束当前代码块。
for (const type of ['dragleave', 'drop']) {  // 监听文件离开或松手放下的事件。
  els.dropzone.addEventListener(type, (event) => {  // 注册对应的拖放事件处理函数。
    event.preventDefault();  // 阻止浏览器执行默认操作。
    els.dropzone.classList.remove('dragover');  // 移除拖放高亮。
  });  // 结束本次调用。
}  // 结束当前代码块。
els.dropzone.addEventListener('drop', (event) => {  // 接收用户放下的文件。
  uploadFiles(event.dataTransfer?.files);  // 将拖放事件中的文件交给上传流程。
});  // 结束本次调用。

// 对话与来源展示。

function addTurn(question) {  // 创建一轮对话所需的页面元素。
  const empty = els.messages.querySelector('.empty-state');  // 查找首次提问前的提示区域。
  if (empty) empty.remove();  // 开始对话时移除空状态。

  const turn = el('div', 'turn');  // 创建本轮对话容器。
  const q = el('div', 'turn-question');  // 创建问题气泡。
  q.append(el('span', '', question));  // 以纯文本方式显示用户问题。
  turn.append(q);  // 将问题加入本轮对话。

  const answer = el('div', 'turn-answer streaming');  // 为正在生成的回答创建气泡。
  const body = el('div', 'answer-body');  // 创建回答正文容器。
  answer.append(body);  // 将正文容器放入回答气泡。
  turn.append(answer);  // 将回答加入本轮对话。

  els.messages.append(turn);  // 在消息列表末尾显示本轮对话。
  els.messages.scrollTop = els.messages.scrollHeight;  // 滚动到底部以显示最新内容。
  return { turn, answer, body };  // 返回后续渲染要更新的节点。
}  // 结束当前代码块。

// 摘要使用纯文本，去除标题行以避免和来源的标题路径重复。
function snippetText(text) {  // 从命中片段提取适合列表展示的纯文本摘要。
  return text  // 以输入文本为起点依次替换。
    .replace(/^#{1,6}\s+.*$/gm, '')  // 移除独占一行的标题，以免重复标题路径。
    .replace(/\n{2,}/g, ' ')  // 把连续换行合并为空格。
    .replace(/\s+/g, ' ')  // 将其余连续空白压缩为一个空格。
    .trim();  // 去掉摘要两端的空白。
}  // 结束当前代码块。

function renderSources(container, data) {  // 渲染检索来源和命中位置。
  if (!data.sources || data.sources.length === 0) {  // 没有来源时省略来源列表。
    // 回答正文已经解释无内容的原因，省略重复提示。
    return;  // 结束当前处理。
  }  // 结束当前代码块。

  const block = el('div', 'sources');  // 创建来源区域。
  block.append(el('div', 'sources-title', '来源'));  // 添加来源标题。
  const list = el('ul', 'source-list');  // 创建来源列表。

  for (const source of data.sources) {  // 依次处理各条引用。
    const item = el('li', 'source-item');  // 创建单条来源记录。
    const head = el('div', 'source-head');  // 创建来源标题行。
    head.append(el('span', 'source-index', `[${source.index}]`));  // 显示与正文对应的引用编号。
    head.append(el('span', 'source-name', source.source));  // 显示来源文件名。
    const where = [];  // 收集可用的定位信息。
    if (source.page) where.push(`第 ${source.page} 页`);  // 有页码时加入所在页。
    if (source.breadcrumb) where.push(source.breadcrumb);  // 有标题路径时加入所属章节。
    if (source.score !== null && source.score !== undefined) where.push(`相关度 ${source.score}`);  // 有重排分数时加入相关度。
    if (where.length) head.append(el('span', 'source-where', where.join(' · ')));  // 用间隔点连接各项定位信息。
    item.append(head);  // 将标题行加入当前项目。

    if (source.matched_text) {  // 有命中文本时展示摘要。
      item.append(el('p', 'source-snippet', snippetText(source.matched_text)));  // 清理命中文本后放入摘要段落。
    }  // 结束当前代码块。
    list.append(item);  // 将当前来源加入列表。
  }  // 结束当前代码块。
  block.append(list);  // 将全部来源放入来源区域。

  if (data.rerank_error) {  // 检查重排是否降级。
    block.append(el('p', 'notice', `重排已跳过：${data.rerank_error}`));  // 展示跳过重排的原因。
  }  // 结束当前代码块。
  container.append(block);  // 把来源区域加入当前回答。
}  // 结束当前代码块。

function renderDebug(container, data) {  // 展示各阶段的检索诊断信息。
  const details = el('details', 'debug');  // 创建可展开的诊断区域。
  details.append(el('summary', '', '检索调试'));  // 设置折叠区域的标题。

  const stages = el('div', 'debug-stages');  // 创建各阶段的展示容器。
  for (const stage of ['dense', 'sparse', 'fused', 'reranked']) {  // 依次处理向量、关键词、融合和重排结果。
    const block = el('div', 'debug-stage');  // 创建当前阶段的分区。
    block.append(el('h4', '', stage));  // 显示当前检索阶段。
    const table = el('table');  // 创建候选结果表。
    const head = el('tr');  // 创建表头行。
    for (const label of ['id', 'channels', 'rank', 'score']) head.append(el('th', '', label));  // 建立标识、通道、排名和分数四列。
    table.append(head);  // 将表头加入表格。
    for (const row of (data[stage] || []).slice(0, 6)) {  // 每个阶段最多展示前六项。
      const tr = el('tr');  // 创建候选项所在行。
      tr.append(el('td', '', String(row.id).slice(-8)));  // 显示标识末八位以节省空间。
      tr.append(el('td', '', (row.channels || []).join('+')));  // 显示命中该候选的各通道。
      const rank = stage === 'fused' ? row.fused_rank : stage === 'reranked' ? row.rerank_rank : stage === 'dense' ? row.dense_rank : row.sparse_rank;  // 根据当前阶段选择对应的排名字段。
      tr.append(el('td', '', String(rank ?? '－')));  // 显示名次，缺失时用横线占位。
      const score = stage === 'reranked' ? row.rerank_score : row.fused_score;  // 重排阶段展示重排分数，其余取融合分数。
      tr.append(el('td', '', score === null || score === undefined ? '－' : Number(score).toFixed(4)));  // 将有效分数保留四位小数。
      table.append(tr);  // 将候选行加入表格。
    }  // 结束当前代码块。
    block.append(table);  // 将表格加入当前阶段。
    stages.append(block);  // 将阶段分区加入布局。
  }  // 结束当前代码块。
  details.append(stages);  // 把所有阶段放入折叠面板。
  container.append(details);  // 在回答下方展示调试面板。
}  // 结束当前代码块。

async function ask(question) {  // 发送问题并逐帧接收回答。
  if (busy) return;  // 前一个问题仍在处理时忽略重复提交。
  busy = true;  // 标记请求进行中。
  els.send.disabled = true;  // 请求期间禁用发送按钮。
  els.question.disabled = true;  // 请求期间禁用问题输入。

  const { turn, answer, body } = addTurn(question);  // 在页面上为新问题创建回答位置。
  let answerText = '';  // 从空字符串开始累积回答。
  let gotSources = false;  // 记录是否已经收到来源。
  let notes = null;  // 暂时还没有状态提示节点。

  const note = (message, isError = false) => {  // 创建或更新回答旁的状态提示。
    if (!notes) {  // 首次提示时新建节点。
      notes = el('p', `notice${isError ? ' error' : ''}`, message);  // 按是否为错误选择提示样式并填入消息。
      answer.append(notes);  // 把提示放入当前回答。
    } else {  // 处理已有状态或备用分支。
      notes.textContent = message;  // 替换提示文字。
      notes.classList.toggle('error', isError);  // 按本次状态切换错误样式。
    }  // 结束当前代码块。
  };  // 结束定义。

  try {  // 尝试执行本次操作。
    const response = await fetch('/api/chat', {  // 向问答接口发送请求。
      method: 'POST',  // 使用提交请求方法。
      headers: { 'Content-Type': 'application/json' },  // 声明请求体使用结构化数据格式。
      body: JSON.stringify({  // 将提问参数编码为请求正文。
        question,  // 包含当前问题。
        history,  // 附带客户端保留的历史问答。
        debug: els.debugCheck.checked,  // 传递用户选中的调试开关。
      }),  // 结束当前参数。
    });  // 结束本次调用。

    if (!response.ok || !response.body) {  // 检查响应是否成功且包含可读数据流。
      const detail = await response.text();  // 读取服务端返回的错误详情。
      throw new Error(detail || `HTTP ${response.status}`);  // 优先展示响应正文中的原因，缺失时使用状态码。
    }  // 结束当前代码块。

    const reader = response.body.getReader();  // 取得响应数据流的读取器。
    const decoder = new TextDecoder();  // 创建支持跨数据块解码的文本解码器。
    let buffer = '';  // 缓存尚未拼成完整事件的数据。

    while (true) {  // 持续读取直至响应流结束。
      const { done, value } = await reader.read();  // 取出下一批字节及结束标记。
      if (done) break;  // 读到流末尾就退出。
      buffer += decoder.decode(value, { stream: true });  // 保留解码状态并把新文本接到缓冲区。

      let boundary = buffer.indexOf('\n\n');  // 查找标记事件结尾的空行。
      while (boundary !== -1) {  // 缓冲区含完整事件时逐帧处理。
        const frame = buffer.slice(0, boundary);  // 取出分隔符之前的完整事件。
        buffer = buffer.slice(boundary + 2);  // 从缓冲区移除已处理的事件及分隔符。
        boundary = buffer.indexOf('\n\n');  // 继续查找下一个完整事件。

        const parsed = parseFrame(frame);  // 解析当前帧。
        if (!parsed) continue;  // 跳过无法解析的帧。
        const { event, data } = parsed;  // 取出事件名和载荷。

        if (event === 'status') {  // 处理阶段更新通知。
          note(data.stage === 'rewriting' ? '正在理解问题…' : '正在检索…');  // 根据当前阶段显示改写或检索提示。
        } else if (event === 'rewritten') {  // 处理问题改写通知。
          note(`检索用问题：${data.query}`);  // 展示实际用于检索的问题。
        } else if (event === 'sources') {  // 处理检索来源通知。
          gotSources = true;  // 标记服务端已返回来源。
          if (notes) { notes.remove(); notes = null; }  // 移除检索期间的临时提示。
          renderSources(answer, data);  // 在回答下方展示来源。
        } else if (event === 'token') {  // 处理新增的回答片段。
          answerText += data.text;  // 将本段文字追加到完整回答。
          body.innerHTML = renderMarkdown(answerText);  // 根据最新累计内容刷新正文。
          els.messages.scrollTop = els.messages.scrollHeight;  // 滚动到底部以显示最新内容。
        } else if (event === 'debug') {  // 处理调试信息通知。
          renderDebug(answer, data);  // 展示各检索阶段的候选结果。
        } else if (event === 'error') {  // 处理服务端报告的故障。
          answer.classList.remove('streaming');  // 关闭正在生成的光标效果。
          note(data.message, true);  // 以错误样式显示服务端消息。
        } else if (event === 'done') {  // 处理正常结束通知。
          answer.classList.remove('streaming');  // 关闭正在生成的光标效果。
          if (data.answer) {  // 最终事件带完整答案时使用该版本。
            answerText = data.answer;  // 用最终答案校准累计文本。
            body.innerHTML = renderMarkdown(answerText);  // 根据最新累计内容刷新正文。
          }  // 结束当前代码块。
          if (data.truncated) note('回答被截断：生成中断。', true);  // 提醒用户回答曾中断。
          if (!gotSources && !data.empty) note('本次没有命中任何来源。');  // 在预期有结果却未收到来源时提示。
        }  // 结束当前代码块。
      }  // 结束当前代码块。
    }  // 结束当前代码块。
  } catch (error) {  // 处理操作过程中的异常。
    answer.classList.remove('streaming');  // 关闭正在生成的光标效果。
    note(`请求失败：${error.message}`, true);  // 展示问答请求失败的原因。
  } finally {  // 无论请求成功或失败都执行收尾操作。
    if (answerText && !answer.classList.contains('streaming')) {  // 有回答且流式显示已经结束时保留本轮历史。
      history.push({ question, answer: answerText });  // 将问题和回答一起写入历史。
      // 服务端逐次接收历史，客户端在此限制窗口长度。
      while (history.length > 6) history.shift();  // 只保留最近六轮，使客户端与服务端的窗口一致。
    }  // 结束当前代码块。
    busy = false;  // 释放请求中的忙碌标记。
    els.send.disabled = false;  // 恢复发送按钮。
    els.question.disabled = false;  // 恢复问题输入。
    els.question.focus();  // 将焦点移回问题输入框。
  }  // 结束当前代码块。
}  // 结束当前代码块。

els.composer.addEventListener('submit', (event) => {  // 处理提问表单提交。
  event.preventDefault();  // 阻止浏览器执行默认操作。
  const question = els.question.value.trim();  // 读取问题并去除首尾空白。
  if (!question) return;  // 忽略空白问题。
  els.question.value = '';  // 清空已提交的问题。
  els.question.style.height = 'auto';  // 恢复输入框的自然高度。
  ask(question);  // 提交本轮提问。
});  // 结束本次调用。

els.question.addEventListener('keydown', (event) => {  // 监听输入框的键盘事件。
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {  // 仅在普通回车且输入法已完成组词时发送。
    event.preventDefault();  // 阻止浏览器执行默认操作。
    els.composer.requestSubmit();  // 通过表单入口触发提交。
  }  // 结束当前代码块。
});  // 结束本次调用。

els.question.addEventListener('input', () => {  // 输入内容发生变化时调整高度。
  els.question.style.height = 'auto';  // 恢复输入框的自然高度。
  els.question.style.height = `${Math.min(200, els.question.scrollHeight)}px`;  // 按内容高度扩展输入框，最高两百像素。
});  // 结束本次调用。

// 初始化页面。

loadDocuments();  // 进入页面时载入文档列表。
warnIfUnhealthy();  // 启动时检查是否需要提示配置问题。
els.question.focus();  // 将焦点移回问题输入框。