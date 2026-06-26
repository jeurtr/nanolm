document.addEventListener('DOMContentLoaded', () => {
    /* ── 元素引用 ── */
    const chatForm = document.getElementById('chat-form');
    const messageInput = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const chatContainer = document.getElementById('chat-container').querySelector('.max-w-4xl');
    const settingsBtn = document.getElementById('settings-btn');
    const settingsPanel = document.getElementById('settings-panel');

    const temperatureSlider = document.getElementById('temperature-slider');
    const temperatureInput = document.getElementById('temperature-input');
    const topPSlider = document.getElementById('top-p-slider');
    const topPInput = document.getElementById('top-p-input');

    /* ── 状态变量 ── */
    let temperature = parseFloat(temperatureSlider.value);
    let topP = parseFloat(topPSlider.value);
    let isGenerating = false;
    let currentAbortController = null;

    /* ── 初始化参数控制器 ── */
    function initializeParameters() {
        temperatureInput.value = parseFloat(temperatureSlider.value).toFixed(2);
        topPInput.value = parseFloat(topPSlider.value).toFixed(2);

        temperatureSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            temperature = value;
            temperatureInput.value = value.toFixed(2);
        });
        temperatureInput.addEventListener('change', (e) => {
            let value = parseFloat(e.target.value);
            if (isNaN(value)) value = 1.0;
            value = Math.max(0, Math.min(2, value));
            temperature = value;
            temperatureInput.value = value.toFixed(2);
            temperatureSlider.value = value;
        });

        topPSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            topP = value;
            topPInput.value = value.toFixed(2);
        });
        topPInput.addEventListener('change', (e) => {
            let value = parseFloat(e.target.value);
            if (isNaN(value)) value = 0.95;
            value = Math.max(0, Math.min(1, value));
            topP = value;
            topPInput.value = value.toFixed(2);
            topPSlider.value = value;
        });
    }

    /* ── 事件监听 ── */
    messageInput.addEventListener('input', () => {
        updateSendButtonState();
        adjustTextareaHeight();
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const messageText = messageInput.value.trim();
        if (messageText && !isGenerating) {
            await sendMessage(messageText);
        }
    });

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit', { cancelable: true }));
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (!settingsPanel.classList.contains('hidden')) {
                settingsPanel.classList.add('hidden');
            } else if (messageInput.value.trim()) {
                messageInput.value = '';
                messageInput.dispatchEvent(new Event('input'));
            } else {
                messageInput.blur();
            }
        }
    });

    settingsBtn.addEventListener('click', () => {
        settingsPanel.classList.toggle('hidden');
    });

    /* ── 核心函数 ── */
    function updateSendButtonState() {
        if (isGenerating) return;
        sendBtn.disabled = messageInput.value.trim() === '';
    }

    function setGeneratingState(generating) {
        isGenerating = generating;
        if (generating) {
            sendBtn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>`;
            sendBtn.classList.add('stop-btn');
            sendBtn.disabled = false;
        } else {
            sendBtn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-white"><path d="m22 2-7 20-4-9-9-4Z"></path><path d="M22 2 11 13"></path></svg>`;
            sendBtn.classList.remove('stop-btn');
            sendBtn.disabled = true;
        }
    }

    function adjustTextareaHeight() {
        messageInput.style.height = 'auto';
        const maxHeight = 200;
        messageInput.style.height = `${Math.min(messageInput.scrollHeight, maxHeight)}px`;
        messageInput.style.overflowY = messageInput.scrollHeight > maxHeight ? 'auto' : 'hidden';
    }

    async function sendMessage(text) {
        const escaped = text.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
        const userHtml = `
            <div class="flex items-start gap-4 my-6 justify-end message-enter" data-role="user" data-content="${escaped}">
                <div class="bg-blue-500 p-4 rounded-xl rounded-br-none prose prose-invert max-w-[80%] message-bubble">
                    ${marked.parse(text)}
                </div>
                <div class="w-10 h-10 rounded-full bg-gray-600 flex items-center justify-center font-bold flex-shrink-0">你</div>
            </div>`;
        chatContainer.insertAdjacentHTML('beforeend', userHtml);
        scrollToBottom();

        messageInput.value = '';
        messageInput.dispatchEvent(new Event('input'));

        settingsPanel.classList.add('hidden');

        const chatHistory = constructChatHistoryPayload();
        await fetchAIResponseWithPost(chatHistory);
    }

    function constructChatHistoryPayload() {
        const history = [];
        const messages = chatContainer.querySelectorAll('[data-role]');
        messages.forEach(msg => {
            const role = msg.dataset.role;
            const content = msg.dataset.content;
            if (role && content) {
                history.push({ role: role, content: content });
            }
        });
        return history;
    }

    function renderContent(element, text) {
        if (!element || typeof marked === 'undefined' || typeof hljs === 'undefined') {
            if (element) element.textContent = text;
            return;
        }
        if (text.includes('{{assistant_name}}')) {
            text = text.replace('{{assistant_name}}', 'NanoLM');
        }
        element.innerHTML = marked.parse(text);
        element.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
            const lang = block.className.replace('hljs', '').replace('language-', '').trim();
            if (lang && block.parentElement) {
                block.parentElement.classList.add('code-block-wrapper');
                const tag = document.createElement('span');
                tag.className = 'code-lang-tag';
                tag.textContent = lang;
                block.parentElement.appendChild(tag);
            }
        });
        if (typeof renderMathInElement !== 'undefined') {
            renderMathInElement(element, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\(', right: '\\)', display: false },
                    { left: '\\[', right: '\\]', display: true }
                ],
                throwOnError: false
            });
        }
    }

    function createMessageActions(messageId) {
        const actions = document.createElement('div');
        actions.className = 'flex items-center gap-1 mt-2';

        /* 复制按钮 */
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100 transition-colors';
        copyBtn.title = '复制';
        copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
        copyBtn.addEventListener('click', () => {
            const content = document.getElementById(messageId).dataset.content || '';
            navigator.clipboard.writeText(content).then(() => {
                copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="check-pop"><polyline points="20 6 9 17 4 12"/></svg>';
                setTimeout(() => {
                    copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
                }, 2000);
            });
        });
        actions.appendChild(copyBtn);

        /* 重新生成按钮 (仅最后一条 AI 消息) */
        const allAIMessages = chatContainer.querySelectorAll('[data-role="assistant"]');
        if (allAIMessages.length > 0 && allAIMessages[allAIMessages.length - 1].id === messageId) {
            const regenBtn = document.createElement('button');
            regenBtn.className = 'regen-btn p-1.5 rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100 transition-colors';
            regenBtn.title = '重新生成';
            regenBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>';
            regenBtn.addEventListener('click', () => {
                /* 移除最后一条 AI 消息 */
                const aiMsg = document.getElementById(messageId);
                if (aiMsg) aiMsg.remove();
                /* 重新发送 */
                const history = constructChatHistoryPayload();
                /* 移除最后一条 user 的 data-content 对应的历史 */
                fetchAIResponseWithPost(history);
            });
            actions.appendChild(regenBtn);
        }

        return actions;
    }

    async function fetchAIResponseWithPost(chatHistory) {
        const messageId = `ai-message-${Date.now()}`;
        const initialAIHtml = `
            <div id="${messageId}" data-role="assistant" class="ai-message-wrapper flex flex-col message-enter">
                <div class="flex items-start gap-4 my-6">
                    <div class="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white flex-shrink-0">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>
                    </div>
                    <div class="w-full">
                        <div id="typing-indicator-${messageId}" class="bg-white border border-gray-100 p-4 rounded-xl rounded-tl-none">
                            <div class="typing-indicator"><span></span><span></span><span></span></div>
                        </div>
                        <div id="final-answer-container-${messageId}" class="w-full bg-white border border-gray-100 p-4 rounded-xl rounded-tl-none prose max-w-none" style="display: none;">
                            <div id="final-answer-text-${messageId}"></div>
                        </div>
                    </div>
                </div>
            </div>`;
        chatContainer.insertAdjacentHTML('beforeend', initialAIHtml);
        scrollToBottom();

        const aiMessageContainer = document.getElementById(messageId);
        const typingIndicator = document.getElementById(`typing-indicator-${messageId}`);
        const answerContainer = document.getElementById(`final-answer-container-${messageId}`);
        const answerTextEl = document.getElementById(`final-answer-text-${messageId}`);

        let firstChunkReceived = false;
        let contentAccumulator = '';

        setGeneratingState(true);
        currentAbortController = new AbortController();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    history: chatHistory,
                    temperature: temperature,
                    top_p: topP,
                }),
                signal: currentAbortController.signal,
            });

            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            const processStream = async () => {
                while (true) {
                    const { value, done } = await reader.read();
                    if (done) {
                        aiMessageContainer.dataset.content = contentAccumulator;
                        const actionsEl = createMessageActions(messageId);
                        aiMessageContainer.appendChild(actionsEl);
                        break;
                    }
                    buffer += decoder.decode(value, { stream: true });

                    let eomIndex;
                    while ((eomIndex = buffer.indexOf('\n\n')) >= 0) {
                        const message = buffer.slice(0, eomIndex);
                        buffer = buffer.slice(eomIndex + 2);
                        processSSEMessage(message, messageId);
                    }
                }
            };

            await processStream();

        } catch (error) {
            if (error.name === 'AbortError') {
                typingIndicator?.remove();
                contentAccumulator += ' [已停止生成]';
                answerTextEl.innerHTML = `<span class="text-gray-500">${contentAccumulator}</span>`;
                answerContainer.style.display = 'block';
                aiMessageContainer.dataset.content = contentAccumulator;
            } else {
                console.error('Fetch failed:', error);
                typingIndicator?.remove();
                answerTextEl.innerHTML = '<span class="text-red-500">抱歉，连接到AI服务器时出错。请检查后端服务是否正在运行以及网络连接。</span>';
                answerContainer.style.display = 'block';
            }
        } finally {
            setGeneratingState(false);
            currentAbortController = null;
        }

        function processSSEMessage(message, messageId) {
            if (!message.startsWith('{')) return;
            try {
                const parsedMessage = JSON.parse(message);
                handleServerEvent(parsedMessage.event, parsedMessage.data, messageId);
            } catch (e) {
                console.error('Error parsing SSE message:', message, e);
            }
        }

        function handleServerEvent(event, data, messageId) {
            if (!firstChunkReceived) {
                firstChunkReceived = true;
                typingIndicator?.remove();
            }
            if (event === 'answer_chunk') {
                if (answerContainer.style.display === 'none') {
                    answerContainer.style.display = 'block';
                }
                contentAccumulator = data;
                renderContent(answerTextEl, contentAccumulator);
            } else if (event === 'error') {
                console.error('Server error:', data);
                answerTextEl.innerHTML = `<span class="text-red-500">服务器返回错误: ${data}</span>`;
                answerContainer.style.display = 'block';
            }
            scrollToBottom();
        }
    }

    /* ── 发送按钮点击：生成中则停止 ── */
    sendBtn.addEventListener('click', (e) => {
        if (isGenerating) {
            e.preventDefault();
            if (currentAbortController) {
                currentAbortController.abort();
            }
            return;
        }
    });

    function scrollToBottom() {
        chatContainer.parentElement.scrollTop = chatContainer.parentElement.scrollHeight;
    }

    /* ── 初始化 ── */
    initializeParameters();
    updateSendButtonState();
    adjustTextareaHeight();
    messageInput.focus();
});
