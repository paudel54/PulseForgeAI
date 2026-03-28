document.addEventListener('DOMContentLoaded', () => {

    // ---- DOM References ----
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-upload');
    const statusMsg = document.getElementById('upload-status');
    const queryInput = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-btn');
    const chatMessages = document.getElementById('chat-messages');
    const docList = document.getElementById('doc-list');
    const docEmpty = document.getElementById('doc-empty');
    const docCountBadge = document.getElementById('doc-count');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    const modelSelect = document.getElementById('model-select');
    const agentNameEl = document.querySelector('.agent-name');

    // ---- Model Selection ----
    modelSelect.addEventListener('change', () => {
        agentNameEl.textContent = modelSelect.options[modelSelect.selectedIndex].text;
    });

    // ---- Sidebar Tabs ----
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const target = btn.dataset.tab;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(p => p.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(`panel-${target}`).classList.add('active');
            if (target === 'library') loadDocuments();
        });
    });

    // ---- Document Library ----

    async function loadDocuments() {
        try {
            const res = await fetch('/documents');
            const data = await res.json();
            renderDocuments(data.documents || []);
        } catch {
            renderDocuments([]);
        }
    }

    function renderDocuments(docs) {
        const count = docs.length;
        docCountBadge.textContent = count;
        docCountBadge.classList.toggle('hidden', count === 0);
        docEmpty.style.display = count === 0 ? 'flex' : 'none';

        // Remove old document rows (not the empty state)
        docList.querySelectorAll('.doc-item').forEach(el => el.remove());

        docs.forEach(filename => {
            const item = document.createElement('div');
            item.className = 'doc-item fade-in';
            item.innerHTML = `
                <div class="doc-item-info">
                    <i data-lucide="file-text" class="doc-item-icon"></i>
                    <span class="doc-item-name" title="${filename}">${filename}</span>
                </div>
                <button class="doc-delete-btn" title="Remove document" data-filename="${filename}">
                    <i data-lucide="trash-2"></i>
                </button>
            `;
            item.querySelector('.doc-delete-btn').addEventListener('click', () => deleteDocument(filename, item));
            docList.appendChild(item);
        });

        lucide.createIcons();
    }

    async function deleteDocument(filename, itemEl) {
        itemEl.classList.add('deleting');
        try {
            const res = await fetch(`/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' });
            const data = await res.json();
            if (res.ok) {
                itemEl.style.opacity = '0';
                itemEl.style.transform = 'translateX(-10px)';
                setTimeout(() => { itemEl.remove(); loadDocuments(); }, 300);
                addSystemNotice(`🗑️ ${filename} removed from knowledge base.`);
            } else {
                addSystemNotice(`Error: ${data.detail}`, true);
            }
        } catch {
            addSystemNotice('Failed to delete document.', true);
            itemEl.classList.remove('deleting');
        }
    }

    // Initial doc count load
    fetch('/documents')
        .then(r => r.json())
        .then(d => {
            const count = (d.documents || []).length;
            docCountBadge.textContent = count;
            docCountBadge.classList.toggle('hidden', count === 0);
        }).catch(() => { });

    // ---- File Upload ----
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) handleFileUpload(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFileUpload(e.target.files[0]);
    });

    async function handleFileUpload(file) {
        if (file.type !== 'application/pdf') {
            showStatus('Please upload a PDF document.', 'error');
            return;
        }

        showStatus(`Uploading "${file.name}"...`, 'info');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', { method: 'POST', body: formData });
            const result = await response.json();

            if (response.ok) {
                showStatus(`✓ ${result.message}`, 'success');
                // Refresh count badge
                fetch('/documents').then(r => r.json()).then(d => {
                    const count = (d.documents || []).length;
                    docCountBadge.textContent = count;
                    docCountBadge.classList.toggle('hidden', count === 0);
                });
            } else {
                showStatus(result.detail || 'Upload failed.', 'error');
            }
        } catch {
            showStatus('Connection error. Is the server running?', 'error');
        }
    }

    function showStatus(message, type) {
        statusMsg.textContent = message;
        statusMsg.className = `status-msg ${type}`;
        statusMsg.classList.remove('hidden');
        if (type === 'success') {
            setTimeout(() => statusMsg.classList.add('hidden'), 5000);
        }
    }

    // ---- Clear Chat ----
    clearChatBtn.addEventListener('click', () => {
        chatMessages.innerHTML = `
            <div class="message system-message fade-in">
                <div class="avatar system-avatar"><i data-lucide="bot"></i></div>
                <div class="message-content">
                    <p><strong>Conversation cleared.</strong> Ready for new queries.</p>
                </div>
            </div>
        `;
        lucide.createIcons();
    });

    // ---- Chat / Query Logic ----
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });

    // Auto-resize textarea
    queryInput.addEventListener('input', () => {
        queryInput.style.height = 'auto';
        queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + 'px';
    });

    sendBtn.addEventListener('click', sendQuery);

    async function sendQuery() {
        const query = queryInput.value.trim();
        if (!query) return;

        addMessage(query, 'user');
        queryInput.value = '';
        queryInput.style.height = 'auto';

        const loadingId = addLoadingMessage();

        // Polar H10 Dummy Payload
        const patientData = {
            device: "Polar_H10",
            timestamp: new Date().toISOString(),
            metrics: {
                heart_rate_bpm: 72,
                hrv_rmssd_ms: 45.2,
                rr_intervals: [820, 815, 830, 810, 825]
            },
            status: "baseline_resting"
        };

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    patient_data: patientData,
                    model: modelSelect.value
                })
            });

            const result = await response.json();
            document.getElementById(loadingId)?.remove();

            if (response.ok) {
                addMessage(result.llm_response, 'system', result.retrieved_context_preview);
            } else {
                addMessage(`Error: ${result.detail}`, 'system', null, true);
            }
        } catch (error) {
            document.getElementById(loadingId)?.remove();
            addMessage('System Error: Unable to reach the reasoning engine.', 'system', null, true);
        }
    }

    // ---- Message Rendering ----
    function addMessage(text, sender, contextPreview = null, isError = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message fade-in`;

        const icon = sender === 'user' ? 'user' : 'bot';

        let formattedText = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n- /g, '<br>• ');

        const showContext = contextPreview
            && sender === 'system'
            && !contextPreview.startsWith('No relevant');

        msgDiv.innerHTML = `
            <div class="avatar ${sender}-avatar"><i data-lucide="${icon}"></i></div>
            <div class="message-content ${isError ? 'error-content' : ''}">
                <p>${formattedText}</p>
                ${showContext ? `
                <div class="context-preview">
                    <strong>📄 RAG Context:</strong> ${contextPreview}
                </div>` : ''}
            </div>
        `;

        chatMessages.appendChild(msgDiv);
        lucide.createIcons();
        scrollToBottom();
    }

    function addSystemNotice(text, isError = false) {
        const div = document.createElement('div');
        div.className = 'system-notice fade-in';
        div.style.color = isError ? 'var(--danger)' : 'var(--text-secondary)';
        div.textContent = text;
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    function addLoadingMessage() {
        const id = 'loading-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.id = id;
        msgDiv.className = 'message system-message fade-in';
        msgDiv.innerHTML = `
            <div class="avatar system-avatar"><i data-lucide="bot"></i></div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        chatMessages.appendChild(msgDiv);
        lucide.createIcons();
        scrollToBottom();
        return id;
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
