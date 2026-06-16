// ============================================
// Global Variables
// ============================================
const complaintTextarea = document.getElementById('complaintText');
const charCount = document.getElementById('charCount');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const predictBothBtn = document.getElementById('predictBothBtn');
const downloadBtn = document.getElementById('downloadBtn');
const sampleSelect = document.getElementById('sampleSelect');
const targetSelect = document.getElementById('targetSelect');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsSection = document.getElementById('resultsSection');
const singleResultsBlock = document.getElementById('singleResultsBlock');
const dualResultsBlock = document.getElementById('dualResultsBlock');
const ensembleSection = document.getElementById('ensembleSection');
const ensembleCard = document.getElementById('ensembleCard');
const resultsContainer = document.getElementById('resultsContainer');
const productEnsembleCard = document.getElementById('productEnsembleCard');
const productResultsContainer = document.getElementById('productResultsContainer');
const issueEnsembleCard = document.getElementById('issueEnsembleCard');
const issueResultsContainer = document.getElementById('issueResultsContainer');
const uncertaintyBanner = document.getElementById('uncertaintyBanner');
const timestamp = document.getElementById('timestamp');
const modelStatus = document.getElementById('modelStatus');
const voiceBtn = document.getElementById('voiceBtn');

let mediaRecorder = null;
let mediaStream = null;
let audioChunks = [];
let isRecording = false;
let isTranscribing = false;
let lastExportPayload = null;

const SAMPLE_COMPLAINTS = {
    credit_card: {
        target: 'product',
        text: `I have been trying to resolve an issue with my credit card statement for over two months. The charges are incorrect and I have been charged late fees and interest that I should not owe. I called customer service multiple times and was told the dispute was opened, but nothing has changed on my account. I am requesting a full review of the billing cycle and removal of all improper fees.`,
    },
    mortgage: {
        target: 'product',
        text: `We applied for a mortgage refinance with our lender eight weeks ago and were told closing would happen within 30 days. Since then we have submitted the same documents three times and still have no clear timeline. Our rate lock expires soon and we may lose thousands of dollars because of the delay. Please assign someone who can explain what is missing and when we can expect to close.`,
    },
    student_loan: {
        target: 'issue',
        text: `My student loan servicer reported my account as delinquent to the credit bureaus even though I was in an approved income-driven repayment plan and made every payment on time. This incorrect reporting dropped my credit score by nearly 80 points and affected my ability to rent an apartment. I need the servicer to correct the credit reporting immediately and provide documentation that the account is in good standing.`,
    },
    duplicate_charge: {
        target: 'issue',
        text: `I was charged twice for the same monthly subscription on my checking account. I only authorized one payment but see duplicate withdrawals on the same day for identical amounts. I contacted the company by email and received no response. I want a refund for the duplicate charge and confirmation that this will not happen again next billing cycle.`,
    },
    unauthorized_debit: {
        target: 'product',
        text: `There are several unauthorized debit transactions on my checking account from merchants I do not recognize. I did not share my card information with anyone and I believe my account may have been compromised. I reported the transactions to the bank but some were denied for reimbursement. I need all fraudulent charges reversed and a new debit card issued as soon as possible.`,
    },
};

// ============================================
// Initialize
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    updateCharCount();
    loadModelStatus();
    initVoiceInput();
    initSampleSelect();

    complaintTextarea.addEventListener('input', () => {
        updateCharCount();
        if (sampleSelect && sampleSelect.value) {
            sampleSelect.value = '';
        }
    });

    clearBtn.addEventListener('click', clearInput);
    predictBtn.addEventListener('click', handlePredict);
    if (predictBothBtn) {
        predictBothBtn.addEventListener('click', handlePredictBoth);
    }
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadResults);
    }

    complaintTextarea.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            handlePredict();
        }
    });
});

function initSampleSelect() {
    if (!sampleSelect) {
        return;
    }

    sampleSelect.addEventListener('change', () => {
        const key = sampleSelect.value;
        if (!key || !SAMPLE_COMPLAINTS[key]) {
            return;
        }

        const sample = SAMPLE_COMPLAINTS[key];
        complaintTextarea.value = sample.text;
        if (targetSelect && sample.target) {
            targetSelect.value = sample.target;
        }
        updateCharCount();
        complaintTextarea.focus();
    });
}

// ============================================
// Character Count
// ============================================
function updateCharCount() {
    const count = complaintTextarea.value.length;
    charCount.textContent = count.toLocaleString();

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
    if (sampleSelect) {
        sampleSelect.value = '';
    }
    updateCharCount();
    hideResults();
    lastExportPayload = null;
    updateDownloadButton();
    complaintTextarea.focus();
}

function validateComplaintText() {
    const text = complaintTextarea.value.trim();

    if (!text) {
        showError('Please enter complaint text before analyzing.');
        return null;
    }

    if (text.length < 10) {
        showError('Please enter at least 10 characters for accurate prediction.');
        return null;
    }

    return text;
}

// ============================================
// Prediction API
// ============================================
async function fetchPrediction(text, target) {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text, target }),
    });

    const data = await response.json();

    if (!data.success) {
        throw new Error(data.error || 'Prediction failed. Please try again.');
    }

    return data;
}

// ============================================
// Handle Prediction
// ============================================
async function handlePredict() {
    const text = validateComplaintText();
    if (!text) {
        return;
    }

    const target = targetSelect ? targetSelect.value : 'product';

    showLoading('Analyzing complaint with all models (including BERT)...');
    hideResults();
    disableButtons();

    try {
        const data = await fetchPrediction(text, target);
        displaySingleResults(data);
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Network error. Please check your connection and try again.');
    } finally {
        hideLoading();
        enableButtons();
    }
}

async function handlePredictBoth() {
    const text = validateComplaintText();
    if (!text) {
        return;
    }

    showLoading('Analyzing product and issue classification (all models)...');
    hideResults();
    disableButtons();

    try {
        const [productData, issueData] = await Promise.all([
            fetchPrediction(text, 'product'),
            fetchPrediction(text, 'issue'),
        ]);
        displayDualResults(text, productData, issueData);
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Network error. Please check your connection and try again.');
    } finally {
        hideLoading();
        enableButtons();
    }
}

// ============================================
// Display Results
// ============================================
function displaySingleResults(data) {
    singleResultsBlock.style.display = 'block';
    dualResultsBlock.style.display = 'none';

    const target = data.target || 'product';
    const targetLabel = target === 'product' ? 'Product Classification' : 'Issue Classification';

    if (data.timestamp) {
        timestamp.textContent = `Last updated: ${data.timestamp} | ${targetLabel}`;
    }

    const resultsTitle = document.querySelector('.results-title');
    if (resultsTitle) {
        resultsTitle.innerHTML = `<span class="card-icon">📊</span> Prediction Results - ${targetLabel}`;
    }

    if (data.final_ensemble_prediction) {
        renderEnsembleCard(ensembleCard, data.final_ensemble_prediction);
        ensembleSection.style.display = 'block';
    } else {
        ensembleSection.style.display = 'none';
        ensembleCard.innerHTML = '';
    }

    renderModelCards(resultsContainer, data.results);
    updateUncertaintyBanner([data.final_ensemble_prediction]);

    lastExportPayload = {
        mode: 'single',
        timestamp: data.timestamp,
        text: complaintTextarea.value.trim(),
        target,
        final_ensemble_prediction: data.final_ensemble_prediction,
        results: data.results,
    };
    updateDownloadButton();

    resultsSection.style.display = 'block';
    scrollToResults();
}

function displayDualResults(text, productData, issueData) {
    singleResultsBlock.style.display = 'none';
    dualResultsBlock.style.display = 'block';
    ensembleSection.style.display = 'none';

    const ts = productData.timestamp || issueData.timestamp;
    if (ts) {
        timestamp.textContent = `Last updated: ${ts} | Product + Issue`;
    }

    const resultsTitle = document.querySelector('.results-title');
    if (resultsTitle) {
        resultsTitle.innerHTML = '<span class="card-icon">📊</span> Prediction Results - Product & Issue';
    }

    renderEnsembleCard(productEnsembleCard, productData.final_ensemble_prediction);
    renderEnsembleCard(issueEnsembleCard, issueData.final_ensemble_prediction);
    renderModelCards(productResultsContainer, productData.results);
    renderModelCards(issueResultsContainer, issueData.results);

    updateUncertaintyBanner([
        productData.final_ensemble_prediction,
        issueData.final_ensemble_prediction,
    ]);

    lastExportPayload = {
        mode: 'both',
        timestamp: ts,
        text,
        product: {
            target: 'product',
            final_ensemble_prediction: productData.final_ensemble_prediction,
            results: productData.results,
        },
        issue: {
            target: 'issue',
            final_ensemble_prediction: issueData.final_ensemble_prediction,
            results: issueData.results,
        },
    };
    updateDownloadButton();

    resultsSection.style.display = 'block';
    scrollToResults();
}

function renderEnsembleCard(container, ensemble) {
    if (!container) {
        return;
    }

    if (!ensemble) {
        container.innerHTML = '<p class="ensemble-empty">No ensemble prediction available.</p>';
        return;
    }

    container.innerHTML = buildEnsembleCardHtml(ensemble);
}

function buildEnsembleCardHtml(ensemble) {
    const confidencePercent = (ensemble.confidence * 100).toFixed(1);
    const majority = ensemble.majority_vote || {};
    const majorityPercent = majority.total
        ? ((majority.votes / majority.total) * 100).toFixed(0)
        : '0';

    return `
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

function renderModelCards(container, results) {
    if (!container) {
        return;
    }

    container.innerHTML = '';

    for (const [modelName, result] of Object.entries(results || {})) {
        if (result.error) {
            createErrorCard(container, modelName, result.error);
        } else {
            createResultCard(container, modelName, result);
        }
    }
}

function updateUncertaintyBanner(ensembles) {
    if (!uncertaintyBanner) {
        return;
    }

    const warnings = (ensembles || [])
        .filter(Boolean)
        .map(getUncertaintyMessage)
        .filter(Boolean);

    if (warnings.length === 0) {
        uncertaintyBanner.style.display = 'none';
        uncertaintyBanner.innerHTML = '';
        return;
    }

    uncertaintyBanner.style.display = 'block';
    uncertaintyBanner.innerHTML = `
        <div class="uncertainty-title">Low confidence — review recommended</div>
        <ul class="uncertainty-list">
            ${warnings.map((msg) => `<li>${escapeHtml(msg)}</li>`).join('')}
        </ul>
    `;
}

function getUncertaintyMessage(ensemble) {
    if (!ensemble) {
        return null;
    }

    const reasons = [];
    const confidence = ensemble.confidence ?? 0;
    const majority = ensemble.majority_vote || {};

    if (confidence < 0.5) {
        reasons.push(`ensemble confidence is ${(confidence * 100).toFixed(1)}%`);
    }

    if (majority.total && majority.votes < 4) {
        reasons.push(`only ${majority.votes}/${majority.total} models agreed on "${ensemble.prediction}"`);
    }

    if (reasons.length === 0) {
        return null;
    }

    return `${ensemble.prediction}: ${reasons.join('; ')}`;
}

function scrollToResults() {
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        animateConfidenceBars();
    }, 100);
}

// ============================================
// Create Result Card
// ============================================
function createResultCard(container, modelName, result) {
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

    container.appendChild(card);
}

function createErrorCard(container, modelName, error) {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.style.borderLeftColor = 'var(--danger-color)';

    card.innerHTML = `
        <div class="model-name" style="color: var(--danger-color);">
            <span>⚠️</span>
            ${modelName}
        </div>
        <div class="model-error-text">
            ${escapeHtml(error)}
        </div>
    `;

    container.appendChild(card);
}

// ============================================
// Download Results
// ============================================
function updateDownloadButton() {
    if (!downloadBtn) {
        return;
    }
    downloadBtn.style.display = lastExportPayload ? 'inline-flex' : 'none';
}

function downloadResults() {
    if (!lastExportPayload) {
        showError('No results to download yet. Run an analysis first.');
        return;
    }

    const stamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
    const blob = new Blob([JSON.stringify(lastExportPayload, null, 2)], {
        type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `complaint-prediction-${stamp}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}

// ============================================
// Animate Confidence Bars
// ============================================
function animateConfidenceBars() {
    const bars = document.querySelectorAll('.confidence-fill');
    bars.forEach((bar) => {
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
    if (uncertaintyBanner) {
        uncertaintyBanner.style.display = 'none';
        uncertaintyBanner.innerHTML = '';
    }
    if (downloadBtn) {
        downloadBtn.style.display = 'none';
    }
}

// ============================================
// Loading States
// ============================================
function showLoading(message) {
    if (loadingIndicator) {
        const textNode = loadingIndicator.querySelector('p');
        if (textNode && message) {
            textNode.textContent = message;
        }
        loadingIndicator.style.display = 'block';
        loadingIndicator.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

function hideLoading() {
    if (loadingIndicator) {
        loadingIndicator.style.display = 'none';
    }
}

function disableButtons() {
    predictBtn.disabled = true;
    clearBtn.disabled = true;
    if (predictBothBtn) {
        predictBothBtn.disabled = true;
    }
    if (sampleSelect) {
        sampleSelect.disabled = true;
    }
    if (voiceBtn && !isRecording) {
        voiceBtn.disabled = true;
    }
}

function enableButtons() {
    if (isRecording || isTranscribing) {
        return;
    }

    predictBtn.disabled = false;
    clearBtn.disabled = false;
    if (predictBothBtn) {
        predictBothBtn.disabled = false;
    }
    if (sampleSelect) {
        sampleSelect.disabled = false;
    }
    if (voiceBtn) {
        voiceBtn.disabled = false;
    }
    loadModelStatus();
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
                            ${status.vectorizer_loaded ? 'Loaded' : 'Not Loaded'}
                        </span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Label Encoder</span>
                        <span class="status-badge ${status.label_encoder_loaded ? 'success' : 'error'}">
                            ${status.label_encoder_loaded ? 'Loaded' : 'Not Loaded'}
                        </span>
                    </div>
                    ${status.bert_loaded !== undefined ? `
                    <div class="status-item">
                        <span class="status-label">BERT Model</span>
                        <span class="status-badge ${status.bert_loaded ? 'success' : 'error'}">
                            ${status.bert_loaded ? 'Loaded' : 'Not Loaded'}
                        </span>
                    </div>
                    ` : ''}
                    ${status.model_names && status.model_names.length > 0 ? `
                    <div class="status-item" style="flex-direction: column; align-items: flex-start; gap: 10px;">
                        <span class="status-label">Available Models:</span>
                        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                            ${status.model_names.map((name) => {
                                const isBert = name === 'BERT';
                                return `<span class="status-badge ${isBert ? 'info' : 'success'}" style="${isBert ? 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);' : ''}">${escapeHtml(name)}</span>`;
                            }).join('')}
                        </div>
                    </div>
                    ` : ''}
                </div>
            `;
        }
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
        if (voiceBtn && !isRecording && !isTranscribing) {
            voiceBtn.disabled = !voiceAvailable;
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
    const existingToasts = document.querySelectorAll('.toast-error');
    existingToasts.forEach((err) => err.remove());

    const errorDiv = document.createElement('div');
    errorDiv.className = 'toast-error';
    errorDiv.textContent = message;

    const inputSection = document.querySelector('.input-section');
    inputSection.insertAdjacentElement('afterend', errorDiv);
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });

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
            enableButtons();
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
        disableButtons();
        voiceBtn.disabled = false;
    } catch (error) {
        console.error('Microphone error:', error);
        cleanupMediaStream();
        isRecording = false;
        resetVoiceButton();
        enableButtons();
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
    isTranscribing = true;
    predictBtn.disabled = true;
    if (predictBothBtn) {
        predictBothBtn.disabled = true;
    }
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
            if (sampleSelect) {
                sampleSelect.value = '';
            }
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
        isRecording = false;
        resetVoiceButton();
        enableButtons();
    }
}
