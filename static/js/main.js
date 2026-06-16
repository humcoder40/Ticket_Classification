// ============================================
// Global Variables
// ============================================
const complaintTextarea = document.getElementById('complaintText');
const charCount = document.getElementById('charCount');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsSection = document.getElementById('resultsSection');
const ensembleSection = document.getElementById('ensembleSection');
const ensembleCard = document.getElementById('ensembleCard');
const resultsContainer = document.getElementById('resultsContainer');
const timestamp = document.getElementById('timestamp');
const modelStatus = document.getElementById('modelStatus');
const voiceBtn = document.getElementById('voiceBtn');
let mediaRecorder = null;
let mediaStream = null;
let audioChunks = [];
let isRecording = false;
let isTranscribing = false;

// ============================================
// Initialize
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    updateCharCount();
    loadModelStatus();
    initVoiceInput();
    
    // Update char count as user types
    complaintTextarea.addEventListener('input', updateCharCount);
    
    // Clear button
    clearBtn.addEventListener('click', clearInput);
    
    // Predict button
    predictBtn.addEventListener('click', handlePredict);
    
    // Enter key shortcut (Ctrl+Enter to predict)
    complaintTextarea.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            handlePredict();
        }
    });
});

// ============================================
// Character Count
// ============================================
function updateCharCount() {
    const count = complaintTextarea.value.length;
    charCount.textContent = count.toLocaleString();
    
    // Change color based on length
    if (count > 5000) {
        charCount.style.color = 'var(--warning-color)';
    } else if (count > 2000) {
        charCount.style.color = 'var(--info-color)';
    } else {
        charCount.style.color = 'var(--text-muted)';
    }
}

// ============================================
// Clear Input
// ============================================
function clearInput() {
    complaintTextarea.value = '';
    updateCharCount();
    resultsSection.style.display = 'none';
    complaintTextarea.focus();
}

// ============================================
// Handle Prediction
// ============================================
async function handlePredict() {
    const text = complaintTextarea.value.trim();
    const targetSelect = document.getElementById('targetSelect');
    const target = targetSelect ? targetSelect.value : 'product';
    
    if (!text) {
        showError('Please enter complaint text before analyzing.');
        return;
    }
    
    if (text.length < 10) {
        showError('Please enter at least 10 characters for accurate prediction.');
        return;
    }
    
    // Show loading
    showLoading();
    hideResults();
    disableButtons();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text, target: target })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data.results, data.timestamp, data.target, data.final_ensemble_prediction);
        } else {
            showError(data.error || 'Prediction failed. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Network error. Please check your connection and try again.');
    } finally {
        hideLoading();
        enableButtons();
    }
}

// ============================================
// Display Results
// ============================================
function displayResults(results, ts, target = 'product', finalEnsemble = null) {
    resultsContainer.innerHTML = '';
    
    // Update timestamp and target info
    const targetLabel = target === 'product' ? 'Product Classification' : 'Issue Classification';
    if (ts) {
        timestamp.textContent = `Last updated: ${ts} | ${targetLabel}`;
    }
    
    // Update results title to show target
    const resultsTitle = document.querySelector('.results-title');
    if (resultsTitle) {
        resultsTitle.innerHTML = `<span class="card-icon">📊</span> Prediction Results - ${targetLabel}`;
    }

    if (finalEnsemble) {
        displayFinalEnsemble(finalEnsemble);
        ensembleSection.style.display = 'block';
    } else {
        ensembleSection.style.display = 'none';
        ensembleCard.innerHTML = '';
    }
    
    // Create result cards for each model
    for (const [modelName, result] of Object.entries(results)) {
        if (result.error) {
            createErrorCard(modelName, result.error);
        } else {
            createResultCard(modelName, result);
        }
    }
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Smooth scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
    
    // Animate confidence bars
    setTimeout(() => {
        animateConfidenceBars();
    }, 200);
}

function displayFinalEnsemble(ensemble) {
    const confidencePercent = (ensemble.confidence * 100).toFixed(1);
    const majority = ensemble.majority_vote || {};
    const majorityPercent = majority.total
        ? ((majority.votes / majority.total) * 100).toFixed(0)
        : '0';

    ensembleCard.innerHTML = `
        <div class="ensemble-badge">Final Ensemble Prediction</div>
        <div class="ensemble-method">${escapeHtml(ensemble.method === 'soft_voting' ? 'Soft voting (averaged confidence scores)' : ensemble.method)}</div>
        <div class="ensemble-prediction-label">Predicted Category</div>
        <div class="ensemble-prediction-value">${escapeHtml(ensemble.prediction)}</div>
        <div class="confidence-bar ensemble-confidence-bar">
            <div class="confidence-fill ensemble-confidence-fill" style="width: ${confidencePercent}%"></div>
        </div>
        <div class="ensemble-confidence-text">Ensemble Confidence: ${confidencePercent}%</div>
        <div class="ensemble-meta">
            <span>${ensemble.model_count} model(s): ${escapeHtml((ensemble.models_used || []).join(', '))}</span>
            ${majority.prediction ? `
            <span class="ensemble-majority">
                Majority vote: ${escapeHtml(majority.prediction)} (${majority.votes}/${majority.total}, ${majorityPercent}%)
            </span>` : ''}
        </div>
        ${ensemble.top_3 ? `
        <div class="top-predictions ensemble-top-predictions">
            <div class="top-predictions-title">Ensemble Top 3</div>
            ${ensemble.top_3.map((pred, idx) => `
                <div class="prediction-item">
                    <span class="prediction-class">${idx + 1}. ${escapeHtml(pred.class)}</span>
                    <span class="prediction-prob">${(pred.probability * 100).toFixed(1)}%</span>
                </div>
            `).join('')}
        </div>
        ` : ''}
    `;
}

// ============================================
// Create Result Card
// ============================================
function createResultCard(modelName, result) {
    const card = document.createElement('div');
    card.className = 'result-card';
    
    const confidencePercent = (result.confidence * 100).toFixed(1);
    
    card.innerHTML = `
        <div class="model-name">
            <span>🤖</span>
            ${modelName}
        </div>
        <div class="prediction-info">
            <div class="prediction-label">Predicted Category:</div>
            <div class="prediction-value">${escapeHtml(result.prediction)}</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
            </div>
            <div class="confidence-text">Confidence: ${confidencePercent}%</div>
        </div>
        ${result.top_3 ? `
        <div class="top-predictions">
            <div class="top-predictions-title">Top 3 Predictions</div>
            ${result.top_3.map((pred, idx) => `
                <div class="prediction-item">
                    <span class="prediction-class">${idx + 1}. ${escapeHtml(pred.class)}</span>
                    <span class="prediction-prob">${(pred.probability * 100).toFixed(1)}%</span>
                </div>
            `).join('')}
        </div>
        ` : ''}
    `;
    
    resultsContainer.appendChild(card);
}

// ============================================
// Create Error Card
// ============================================
function createErrorCard(modelName, error) {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.style.borderLeftColor = 'var(--danger-color)';
    
    card.innerHTML = `
        <div class="model-name" style="color: var(--danger-color);">
            <span>⚠️</span>
            ${modelName}
        </div>
        <div class="error-message">
            ${escapeHtml(error)}
        </div>
    `;
    
    resultsContainer.appendChild(card);
}

// ============================================
// Animate Confidence Bars
// ============================================
function animateConfidenceBars() {
    const bars = document.querySelectorAll('.confidence-fill');
    bars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0%';
        setTimeout(() => {
            bar.style.width = width;
        }, 100);
    });
}

function hideResults() {
    resultsSection.style.display = 'none';
    if (ensembleSection) {
        ensembleSection.style.display = 'none';
    }
}

// ============================================
// Loading States
// ============================================
function showLoading() {
    loadingIndicator.style.display = 'block';
    loadingIndicator.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function hideLoading() {
    loadingIndicator.style.display = 'none';
}

function showResults() {
    resultsSection.style.display = 'block';
}

function disableButtons() {
    predictBtn.disabled = true;
    clearBtn.disabled = true;
    if (voiceBtn && !isRecording && !isTranscribing) {
        voiceBtn.disabled = true;
    }
}

function enableButtons() {
    predictBtn.disabled = false;
    clearBtn.disabled = false;
    if (voiceBtn && !isRecording && !isTranscribing) {
        voiceBtn.disabled = false;
    }
}

// ============================================
// Load Model Status
// ============================================
async function loadModelStatus() {
    try {
        const response = await fetch('/models/status');
        const data = await response.json();
        
        displayModelStatus(data);
    } catch (error) {
        console.error('Error loading model status:', error);
        modelStatus.innerHTML = `
            <div class="status-item">
                <span class="status-label">Status</span>
                <span class="status-badge error">Error loading status</span>
            </div>
        `;
    }
}

function displayModelStatus(data) {
    let statusHtml = '';
    
    if (data.targets_status) {
        // New format with multiple targets
        for (const [target, status] of Object.entries(data.targets_status)) {
            const targetLabel = target === 'product' ? 'Product' : 'Issue';
            statusHtml += `
                <div class="status-item" style="flex-direction: column; align-items: flex-start; gap: 10px; margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid var(--border-color);">
                    <div style="font-weight: 600; color: var(--primary-color); margin-bottom: 5px;">${targetLabel} Classification</div>
                    <div class="status-item">
                        <span class="status-label">Models Loaded</span>
                        <span class="status-value">${status.models_loaded}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Vectorizer</span>
                        <span class="status-badge ${status.vectorizer_loaded ? 'success' : 'error'}">
                            ${status.vectorizer_loaded ? '✓ Loaded' : '✗ Not Loaded'}
                        </span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Label Encoder</span>
                        <span class="status-badge ${status.label_encoder_loaded ? 'success' : 'error'}">
                            ${status.label_encoder_loaded ? '✓ Loaded' : '✗ Not Loaded'}
                        </span>
                    </div>
                    ${status.bert_loaded !== undefined ? `
                    <div class="status-item">
                        <span class="status-label">BERT Model</span>
                        <span class="status-badge ${status.bert_loaded ? 'success' : 'error'}">
                            ${status.bert_loaded ? '✓ Loaded' : '✗ Not Loaded'}
                        </span>
                    </div>
                    ` : ''}
                    ${status.model_names && status.model_names.length > 0 ? `
                    <div class="status-item" style="flex-direction: column; align-items: flex-start; gap: 10px;">
                        <span class="status-label">Available Models:</span>
                        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                            ${status.model_names.map(name => {
                                const isBert = name === 'BERT';
                                return `<span class="status-badge ${isBert ? 'info' : 'success'}" style="${isBert ? 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);' : ''}">${escapeHtml(name)}</span>`;
                            }).join('')}
                        </div>
                    </div>
                    ` : ''}
                </div>
            `;
        }
    } else {
        // Fallback to old format
        statusHtml = `
            <div class="status-item">
                <span class="status-label">Models Loaded</span>
                <span class="status-value">${data.models_loaded || 0}</span>
            </div>
            <div class="status-item">
                <span class="status-label">Vectorizer</span>
                <span class="status-badge ${data.vectorizer_loaded ? 'success' : 'error'}">
                    ${data.vectorizer_loaded ? '✓ Loaded' : '✗ Not Loaded'}
                </span>
            </div>
            <div class="status-item">
                <span class="status-label">Label Encoder</span>
                <span class="status-badge ${data.label_encoder_loaded ? 'success' : 'error'}">
                    ${data.label_encoder_loaded ? '✓ Loaded' : '✗ Not Loaded'}
                </span>
            </div>
            ${data.model_names && data.model_names.length > 0 ? `
            <div class="status-item" style="flex-direction: column; align-items: flex-start; gap: 10px;">
                <span class="status-label">Available Models:</span>
                <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                    ${data.model_names.map(name => `
                        <span class="status-badge success">${escapeHtml(name)}</span>
                    `).join('')}
                </div>
            </div>
            ` : ''}
        `;
    }

    if (data.transcription) {
        const voiceAvailable = data.transcription.available;
        const voiceModel = data.transcription.model || 'Groq Whisper';
        statusHtml += `
            <div class="status-item" style="margin-top: 10px; padding-top: 15px; border-top: 1px solid var(--border-color);">
                <span class="status-label">Voice (Groq)</span>
                <span class="status-badge ${voiceAvailable ? 'success' : 'error'}">
                    ${voiceAvailable ? `[OK] ${escapeHtml(voiceModel)}` : 'Not configured (GROQ_API_KEY)'}
                </span>
            </div>
        `;
        if (voiceBtn) {
            voiceBtn.disabled = !voiceAvailable || isRecording || isTranscribing;
            voiceBtn.title = voiceAvailable
                ? 'Record complaint with microphone (Groq Whisper)'
                : 'Set GROQ_API_KEY in .env to enable voice input';
        }
    }

    modelStatus.innerHTML = statusHtml;
}

// ============================================
// Error Display
// ============================================
function showError(message) {
    // Remove existing error messages
    const existingErrors = document.querySelectorAll('.error-message');
    existingErrors.forEach(err => err.remove());
    
    // Create error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    
    // Insert after input section
    const inputSection = document.querySelector('.input-section');
    inputSection.insertAdjacentElement('afterend', errorDiv);
    
    // Scroll to error
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    // Remove after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// ============================================
// Utility Functions
// ============================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function initVoiceInput() {
    if (!voiceBtn) {
        return;
    }

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
        voiceBtn.disabled = true;
        voiceBtn.textContent = 'Voice Not Supported';
        voiceBtn.title = 'Microphone recording is not supported in this browser';
        return;
    }

    voiceBtn.addEventListener('click', async () => {
        if (isTranscribing) {
            return;
        }
        if (isRecording) {
            stopVoiceRecording();
        } else {
            await startVoiceRecording();
        }
    });
}

function getPreferredAudioMimeType() {
    const candidates = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/mp4',
        'audio/ogg;codecs=opus',
    ];
    return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || '';
}

async function startVoiceRecording() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mimeType = getPreferredAudioMimeType();
        const options = mimeType ? { mimeType } : undefined;
        mediaRecorder = new MediaRecorder(mediaStream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onerror = () => {
            cleanupMediaStream();
            isRecording = false;
            resetVoiceButton();
            showError('Recording failed. Please try again.');
        };

        mediaRecorder.onstop = () => {
            const recordedMimeType = mediaRecorder.mimeType || mimeType || 'audio/webm';
            cleanupMediaStream();
            transcribeVoiceRecording(recordedMimeType);
        };

        mediaRecorder.start();
        isRecording = true;
        voiceBtn.textContent = 'Stop Recording';
        voiceBtn.classList.add('recording');
        voiceBtn.title = 'Click to stop and transcribe';
    } catch (error) {
        console.error('Microphone error:', error);
        cleanupMediaStream();
        isRecording = false;
        resetVoiceButton();
        showError('Microphone access denied or unavailable.');
    }
}

function stopVoiceRecording() {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
        return;
    }

    isRecording = false;
    voiceBtn.classList.remove('recording');
    voiceBtn.textContent = 'Transcribing...';
    voiceBtn.disabled = true;
    voiceBtn.classList.add('transcribing');
    mediaRecorder.stop();
}

function cleanupMediaStream() {
    if (mediaStream) {
        mediaStream.getTracks().forEach((track) => track.stop());
        mediaStream = null;
    }
}

function resetVoiceButton() {
    if (!voiceBtn) {
        return;
    }
    voiceBtn.classList.remove('recording', 'transcribing');
    voiceBtn.textContent = 'Voice Input';
}

function extensionForMimeType(mimeType) {
    if (mimeType.includes('webm')) return 'webm';
    if (mimeType.includes('ogg')) return 'ogg';
    if (mimeType.includes('mp4') || mimeType.includes('m4a')) return 'm4a';
    if (mimeType.includes('wav')) return 'wav';
    return 'webm';
}

async function transcribeVoiceRecording(mimeType) {
    isTranscribing = true;

    try {
        const blob = new Blob(audioChunks, { type: mimeType });
        audioChunks = [];

        if (!blob.size) {
            showError('No audio captured. Please try again.');
            return;
        }

        const formData = new FormData();
        formData.append('audio', blob, `complaint.${extensionForMimeType(mimeType)}`);

        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();

        if (data.success && data.text) {
            const existing = complaintTextarea.value.trim();
            complaintTextarea.value = existing ? `${existing} ${data.text}` : data.text;
            updateCharCount();
            complaintTextarea.focus();
        } else {
            showError(data.error || 'Transcription failed. Please try again.');
        }
    } catch (error) {
        console.error('Transcription error:', error);
        showError('Could not reach the transcription service.');
    } finally {
        isTranscribing = false;
        resetVoiceButton();
        loadModelStatus();
    }
}