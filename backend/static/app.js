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
    const btnDoc = document.getElementById('btn-doc');
    const btnPat = document.getElementById('btn-pat');
    const btnReport = document.getElementById('btn-report');
    const agentNameEl = document.querySelector('.agent-name');

    let currentRole = 'doctor';
    let currentModel = 'hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M';

    // ---- Role Selection ----
    function setRole(role) {
        currentRole = role;
        if (role === 'doctor') {
            document.body.classList.remove('theme-patient');
            btnDoc.classList.add('active');
            btnPat.classList.remove('active');
            currentModel = 'hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M';
            agentNameEl.textContent = 'MedGemma 4B';
        } else {
            document.body.classList.add('theme-patient');
            btnPat.classList.add('active');
            btnDoc.classList.remove('active');
            currentModel = 'llama3.1:latest';
            agentNameEl.textContent = 'Llama 3.1';
        }
    }

    btnDoc.addEventListener('click', () => setRole('doctor'));
    btnPat.addEventListener('click', () => setRole('patient'));

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
        if (e.dataTransfer.files.length) handleFileUpload(Array.from(e.dataTransfer.files));
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFileUpload(Array.from(e.target.files));
    });

    async function handleFileUpload(files) {
        showStatus(`Uploading ${files.length} document(s)...`, 'info');
        let hasError = false;

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            if (file.type !== 'application/pdf') {
                hasError = true;
                continue;
            }
            const formData = new FormData();
            formData.append('file', file);
            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                if (!response.ok) hasError = true;
            } catch {
                hasError = true;
            }
        }

        if (hasError) {
            showStatus(`Upload finished with some errors (Ensure all files are PDFs).`, 'error');
        } else {
            showStatus(`✓ Successfully uploaded ${files.length} document(s)`, 'success');
        }

        // Refresh count badge
        fetch('/documents').then(r => r.json()).then(d => {
            const count = (d.documents || []).length;
            docCountBadge.textContent = count;
            docCountBadge.classList.toggle('hidden', count === 0);
        });
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

        if (currentRole === 'patient') {
            patientData.history = {
                "5_days_ago": { heart_rate_bpm: 76, hrv_rmssd_ms: 41.0 },
                "15_days_ago": { heart_rate_bpm: 82, hrv_rmssd_ms: 35.5 }
            };
        }

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query,
                    patient_data: patientData,
                    model: currentModel,
                    role: currentRole
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
                <div style="display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 0.9rem;">
                    <div class="typing-indicator">
                        <span></span><span></span><span></span>
                    </div>
                    <em>Generating response...</em>
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

    // ---- HTML-to-PDF Report Generation ----
    btnReport.addEventListener('click', async () => {
        const query = currentRole === 'patient'
            ? "Generate a simple, highly encouraging progress report summarizing my current vitals vs my 5-day and 15-day history."
            : "Generate a formal clinical evaluation report comparing the patient's baseline resting vitals to their 5-day and 15-day historical data. Provide a professional medical assessment.";

        addMessage(`Requesting formal ${currentRole} report...`, 'user');

        const loadingId = addLoadingMessage();

        const patientData = {
            device: "Polar_H10",
            timestamp: new Date().toISOString(),
            metrics: { heart_rate_bpm: 72, hrv_rmssd_ms: 45.2 },
            history: {
                "5_days_ago": { heart_rate_bpm: 76, hrv_rmssd_ms: 41.0 },
                "15_days_ago": { heart_rate_bpm: 82, hrv_rmssd_ms: 35.5 }
            }
        };

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, patient_data: patientData, model: currentModel, role: currentRole })
            });

            const result = await response.json();
            document.getElementById(loadingId)?.remove();

            if (response.ok) {
                const encodedText = encodeURIComponent(result.llm_response);
                const htmlText = result.llm_response + `<br><br><button onclick="downloadReport(this)" data-content="${encodedText}" class="report-btn" style="display:inline-flex; border-color:var(--text); color:var(--text);"><i data-lucide="printer"></i> Print / Save as PDF</button>`;
                addMessage(htmlText, 'system', null);
            } else {
                addMessage(`Error: ${result.detail}`, 'system', null, true);
            }
        } catch {
            document.getElementById(loadingId)?.remove();
            addMessage('System Error.', 'system', null, true);
        }
    });

    window.downloadReport = function (btn) {
        const text = decodeURIComponent(btn.getAttribute('data-content'));
        const printWindow = window.open('', '_blank');
        if (!printWindow) {
            alert("Popup blocked! Please allow popups to generate the PDF report.");
            return;
        }
        printWindow.document.write(`
            <html>
            <head>
                <title>PulseForgeAI Medical Report</title>
                <style>
                    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; padding: 40px; color: #1e293b; max-width: 800px; margin: 0 auto; line-height: 1.6; }
                    h1 { color: #0ea5e9; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; margin-bottom: 30px; }
                    .report-content { white-space: pre-wrap; margin-top: 20px; font-size: 15px; background: #f8fafc; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0; }
                    table { width: 100%; border-collapse: collapse; margin-top: 20px; margin-bottom: 30px; }
                    th, td { border: 1px solid #cbd5e1; padding: 12px; text-align: left; }
                    th { background-color: #f1f5f9; font-weight: 600; }
                    .footer { margin-top: 50px; font-size: 12px; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 20px; }
                    @media print { body { padding: 0; } .footer { position: fixed; bottom: 0; width: 100%; } }
                </style>
            </head>
            <body>
                <h1>PulseForgeAI Vitals Report</h1>
                <p><strong>Device:</strong> Polar H10 (Simulated Data)</p>
                <table>
                    <tr><th>Timeframe</th><th>Heart Rate (bpm)</th><th>HRV RMSSD (ms)</th></tr>
                    <tr><td>Latest Baseline</td><td>72</td><td>45.2</td></tr>
                    <tr><td>5 Days Ago</td><td>76</td><td>41.0</td></tr>
                    <tr><td>15 Days Ago</td><td>82</td><td>35.5</td></tr>
                </table>
                <h3>AI Clinical Summary / AI Insights</h3>
                <div class="report-content">${text}</div>
                <div class="footer">Generated securely and entirely offline via PulseForgeAI • ${new Date().toLocaleDateString()}</div>
                <script>
                    window.onload = () => { window.print(); };
                </script>
            </body>
            </html>
        `);
        printWindow.document.close();
    };

});
