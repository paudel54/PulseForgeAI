document.addEventListener('DOMContentLoaded', () => {
    // ---- DOM Elements ----
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-upload');
    const statusMsg = document.getElementById('upload-status');
    
    const queryInput = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-btn');
    const chatMessages = document.getElementById('chat-messages');

    // ---- File Upload Logic (Knowledge Base) ----
    
    // Clicking the drop zone opens file dialog
    dropZone.addEventListener('click', () => fileInput.click());

    // Drag and drop visual feedback
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFileUpload(e.target.files[0]);
        }
    });

    async function handleFileUpload(file) {
        if (file.type !== 'application/pdf') {
            showStatus('Please upload a PDF document.', 'error');
            return;
        }

        showStatus(`Uploading ${file.name}...`, 'info');
        statusMsg.className = 'status-msg'; // Reset colors
        
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                showStatus(result.message, 'success');
            } else {
                showStatus(result.detail || 'Upload failed', 'error');
            }
        } catch (error) {
            showStatus('Connection error. Is the server running?', 'error');
        }
    }

    function showStatus(message, type) {
        statusMsg.textContent = message;
        statusMsg.className = `status-msg ${type}`;
        statusMsg.classList.remove('hidden');
        
        if (type === 'success') {
            setTimeout(() => {
                statusMsg.classList.add('hidden');
            }, 5000);
        }
    }

    // ---- Chat Logic (RAG & LLM) ----
    
    // Allow sending message with Enter key
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });

    sendBtn.addEventListener('click', sendQuery);

    async function sendQuery() {
        const query = queryInput.value.trim();
        if (!query) return;

        // 1. Add user message to UI
        addMessage(query, 'user');
        queryInput.value = '';
        
        // 2. Add loading indicator
        const loadingId = addLoadingMessage();

        // 3. Simulated Polar H10 Data Payload
        const patientData = {
            "device": "Polar_H10",
            "timestamp": new Date().toISOString(),
            "metrics": {
                "heart_rate_bpm": 72,
                "hrv_rmssd_ms": 45.2,
                "rr_intervals": [820, 815, 830, 810, 825]
            },
            "status": "baseline_resting"
        };

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, patient_data: patientData })
            });

            const result = await response.json();
            
            // Remove loading indicator
            document.getElementById(loadingId).remove();

            if (response.ok) {
                // Add the LLM response to UI
                addMessage(result.llm_response, 'system', result.retrieved_context_preview);
            } else {
                addMessage(`Error: ${result.detail}`, 'system', null, true);
            }
        } catch (error) {
            document.getElementById(loadingId).remove();
            addMessage('System Error: Unable to reach the reasoning engine.', 'system', null, true);
        }
    }

    function addMessage(text, sender, contextPreview = null, isError = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message fade-in`;
        
        const avatarIcon = sender === 'user' ? 'user' : 'bot';
        
        // Simple Markdown parsing for bullet points and bold text
        let formattedText = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n- /g, '<br>• ');

        let html = `
            <div class="avatar"><i data-lucide="${avatarIcon}"></i></div>
            <div class="message-content">
                <p style="${isError ? 'color: var(--danger);' : ''}">${formattedText}</p>
        `;

        if (contextPreview && sender === 'system') {
            html += `
                <div class="context-preview">
                    <strong>RAG Context Used:</strong> ${contextPreview}
                </div>
            `;
        }

        html += `</div>`;
        msgDiv.innerHTML = html;
        
        chatMessages.appendChild(msgDiv);
        lucide.createIcons(); // Re-initialize icons for new elements
        scrollToBottom();
    }

    function addLoadingMessage() {
        const id = 'loading-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.id = id;
        msgDiv.className = 'message system-message fade-in';
        msgDiv.innerHTML = `
            <div class="avatar"><i data-lucide="bot"></i></div>
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
