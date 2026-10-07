/**
 * Pearson VUE Random Forest Student Classification UI Logic
 * Handles real-time inference, slider sync, batch CSV evaluation,
 * and Random Forest model diagnostics.
 */

let scoredCsvContent = null;
let debounceTimer = null;

document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupSync();
  fetchModelMetrics();
  triggerPrediction();
});

// ==========================================
// 1. Navigation Tabs
// ==========================================
function setupTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  const panes = document.querySelectorAll('.tab-pane');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-tab');

      tabs.forEach(t => t.classList.remove('active'));
      panes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');
    });
  });
}

// ==========================================
// 2. Input Synchronization & Debounce
// ==========================================
function setupSync() {
  const pairs = [
    { slider: 'mstInput', num: 'mstNum' },
    { slider: 'quizInput', num: 'quizNum' },
    { slider: 'attInput', num: 'attNum' },
    { slider: 'asgInput', num: 'asgNum' }
  ];

  pairs.forEach(pair => {
    const sEl = document.getElementById(pair.slider);
    const nEl = document.getElementById(pair.num);

    sEl.addEventListener('input', () => {
      nEl.value = sEl.value;
      debouncedPrediction();
    });

    nEl.addEventListener('input', () => {
      sEl.value = nEl.value;
      debouncedPrediction();
    });
  });

  document.getElementById('asgComp').addEventListener('input', debouncedPrediction);
  document.getElementById('studyHours').addEventListener('input', debouncedPrediction);
}

function syncInput(prefix, val) {
  debouncedPrediction();
}

function debouncedPrediction() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    triggerPrediction();
  }, 200);
}

// ==========================================
// 3. Quick Archetype Presets
// ==========================================
const PRESETS = {
  high: {
    exam: 88.0,
    att: 95.0,
    part: 8.5,
    study: 22.0,
    sleep: 7.5,
    screen: 1.5
  },
  avg: {
    exam: 65.0,
    att: 78.0,
    part: 6.0,
    study: 14.0,
    sleep: 7.0,
    screen: 3.5
  },
  atRisk: {
    exam: 38.0,
    att: 52.0,
    part: 3.0,
    study: 5.5,
    sleep: 5.5,
    screen: 6.0
  }
};

function applyPreset(type) {
  const p = PRESETS[type];
  if (!p) return;

  document.getElementById('mstInput').value = p.exam;
  document.getElementById('mstNum').value = p.exam;

  document.getElementById('attInput').value = p.att;
  document.getElementById('attNum').value = p.att;

  document.getElementById('quizInput').value = p.part;
  document.getElementById('quizNum').value = p.part;

  document.getElementById('asgInput').value = p.study;
  document.getElementById('asgNum').value = p.study;

  document.getElementById('asgComp').value = p.sleep;
  document.getElementById('studyHours').value = p.screen;

  triggerPrediction();
}

// ==========================================
// 4. Live Prediction API Call
// ==========================================
async function triggerPrediction() {
  const payload = {
    exam_score: parseFloat(document.getElementById('mstNum').value) || 0,
    attendance_percentage: parseFloat(document.getElementById('attNum').value) || 0,
    class_participation: parseFloat(document.getElementById('quizNum').value) || 0,
    weekly_self_study_hours: parseFloat(document.getElementById('asgNum').value) || 0,
    sleep_hours: parseFloat(document.getElementById('asgComp').value) || 7.0,
    social_media_hours: parseFloat(document.getElementById('studyHours').value) || 2.0,
    netflix_hours: 1.5
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const json = await res.json();
    if (json.status === 'success') {
      updatePredictionDisplay(json.data);
    } else {
      console.error('API Error:', json.message);
      const titleEl = document.getElementById('predictedTitle');
      const urgencyEl = document.getElementById('urgencyBadge');
      if (titleEl) titleEl.innerText = 'Service Initializing...';
      if (urgencyEl) urgencyEl.innerText = json.message || 'API Error';
    }
  } catch (err) {
    console.error('Network Error during prediction call:', err);
  }
}

function updatePredictionDisplay(data) {
  const titleEl = document.getElementById('predictedTitle');
  const ringEl = document.getElementById('categoryRing');
  const confValEl = document.getElementById('confidenceValue');
  const urgencyEl = document.getElementById('urgencyBadge');

  titleEl.innerText = data.predicted_category;
  confValEl.innerText = `${data.confidence}%`;

  let themeColor = 'var(--color-green)';
  let urgencyText = 'Optimal Trajectory';

  if (data.predicted_category === 'Needs Improvement') {
    themeColor = 'var(--color-red)';
    urgencyText = 'Critical Risk: Action Needed';
    document.getElementById('derivedUrgency').innerText = 'High';
    document.getElementById('derivedUrgency').className = 'd-val red-text';
  } else if (data.predicted_category === 'Average Performer') {
    themeColor = 'var(--color-yellow)';
    urgencyText = 'Moderate Performance';
    document.getElementById('derivedUrgency').innerText = 'Moderate';
    document.getElementById('derivedUrgency').className = 'd-val yellow-text';
  } else {
    document.getElementById('derivedUrgency').innerText = 'Low';
    document.getElementById('derivedUrgency').className = 'd-val green-text';
  }

  ringEl.style.backgroundColor = themeColor;
  ringEl.style.boxShadow = `0 0 20px ${themeColor}`;
  urgencyEl.innerText = urgencyText;
  urgencyEl.style.color = themeColor;
  urgencyEl.style.borderColor = themeColor;

  const probs = data.class_probabilities;
  const highProb = probs['High Performer'] || 0;
  const avgProb = probs['Average Performer'] || 0;
  const lowProb = probs['Needs Improvement'] || 0;

  document.getElementById('probHighVal').innerText = `${highProb}%`;
  document.getElementById('barHigh').style.width = `${highProb}%`;

  document.getElementById('probAvgVal').innerText = `${avgProb}%`;
  document.getElementById('barAvg').style.width = `${avgProb}%`;

  document.getElementById('probLowVal').innerText = `${lowProb}%`;
  document.getElementById('barLow').style.width = `${lowProb}%`;

  if (data.derived_metrics) {
    document.getElementById('derivedComposite').innerText = data.derived_metrics.composite_score;
    document.getElementById('derivedEngagement').innerText = data.derived_metrics.engagement_index;
  }

  const recListEl = document.getElementById('recommendationList');
  recListEl.innerHTML = '';
  data.recommendations.forEach(rec => {
    const li = document.createElement('li');
    li.innerText = rec;
    recListEl.appendChild(li);
  });
}

// ==========================================
// 5. Batch CSV Upload & Processing
// ==========================================
async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  const dropzone = document.getElementById('batchDropzone');
  dropzone.style.opacity = '0.5';

  try {
    const res = await fetch('/api/batch-predict', {
      method: 'POST',
      body: formData
    });

    const json = await res.json();
    dropzone.style.opacity = '1';

    if (json.status === 'success') {
      displayBatchResults(json);
    } else {
      alert(`Batch processing error: ${json.message}`);
    }
  } catch (err) {
    dropzone.style.opacity = '1';
    alert(`Failed to upload and process batch: ${err}`);
  }
}

function displayBatchResults(json) {
  scoredCsvContent = json.csv_content;
  const wrapper = document.getElementById('batchResultsWrapper');
  wrapper.style.display = 'block';

  const sum = json.summary;
  document.getElementById('batchTotalCount').innerText = sum.total_students;

  const counts = sum.distribution_counts;
  const pcts = sum.distribution_percentages;

  document.getElementById('batchHighCount').innerText = counts['High Performer'] || 0;
  document.getElementById('batchHighPct').innerText = `${pcts['High Performer'] || 0}%`;

  document.getElementById('batchAvgCount').innerText = counts['Average Performer'] || 0;
  document.getElementById('batchAvgPct').innerText = `${pcts['Average Performer'] || 0}%`;

  document.getElementById('batchLowCount').innerText = counts['Needs Improvement'] || 0;
  document.getElementById('batchLowPct').innerText = `${pcts['Needs Improvement'] || 0}%`;

  const tbody = document.getElementById('batchTableBody');
  tbody.innerHTML = '';

  json.preview.forEach(row => {
    const tr = document.createElement('tr');

    let badgeClass = 'yellow';
    if (row.predicted_category === 'High Performer') badgeClass = 'green';
    if (row.predicted_category === 'Needs Improvement') badgeClass = 'red';

    tr.innerHTML = `
      <td><strong>${row.student_id || 'PV-DEMO'}</strong></td>
      <td>${row.exam_score !== undefined ? row.exam_score : (row.mst_score || 'N/A')}</td>
      <td>${row.attendance_percentage !== undefined ? row.attendance_percentage + '%' : (row.attendance_rate ? row.attendance_rate + '%' : 'N/A')}</td>
      <td>${row.class_participation !== undefined ? row.class_participation : (row.quiz_avg || 'N/A')}</td>
      <td>${row.weekly_self_study_hours !== undefined ? row.weekly_self_study_hours + ' hrs' : 'N/A'}</td>
      <td><span class="table-badge ${badgeClass}">${row.predicted_category}</span></td>
      <td><strong>${row.confidence_score}%</strong></td>
    `;
    tbody.appendChild(tr);
  });

  wrapper.scrollIntoView({ behavior: 'smooth' });
}

function downloadScoredCSV() {
  if (!scoredCsvContent) return;
  const blob = new Blob([scoredCsvContent], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.setAttribute('hidden', '');
  a.setAttribute('href', url);
  a.setAttribute('download', 'pearson_vue_random_forest_scored.csv');
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// Drag & drop support
const dropzone = document.getElementById('batchDropzone');
['dragenter', 'dragover'].forEach(eventName => {
  dropzone.addEventListener(eventName, e => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--brand-accent)';
  }, false);
});
['dragleave', 'drop'].forEach(eventName => {
  dropzone.addEventListener(eventName, e => {
    e.preventDefault();
    dropzone.style.borderColor = 'rgba(56, 189, 248, 0.3)';
  }, false);
});
dropzone.addEventListener('drop', e => {
  const dt = e.dataTransfer;
  const files = dt.files;
  if (files.length > 0) {
    document.getElementById('csvFileInput').files = files;
    handleFileUpload({ target: { files: files } });
  }
});

// ==========================================
// 6. Random Forest Metrics & Optimization
// ==========================================
async function fetchModelMetrics() {
  try {
    const res = await fetch('/api/model-metrics');
    const json = await res.json();
    if (json.status === 'success') {
      populateLeaderboard(json.data);
    }
  } catch (err) {
    console.error('Failed to load metrics:', err);
  }
}

function populateLeaderboard(data) {
  const models = data.model_comparison;
  const champ = data.champion_model;
  const tbody = document.getElementById('leaderboardBody');
  tbody.innerHTML = '';

  if (models[champ]) {
    document.getElementById('headerF1').innerText = `${(models[champ].oob_score * 100).toFixed(1)}%`;
  }

  Object.keys(models).forEach(name => {
    const m = models[name];
    const isChamp = (name === champ);
    const tr = document.createElement('tr');

    tr.innerHTML = `
      <td><strong>${name}</strong></td>
      <td>${(m.cv_accuracy_mean * 100).toFixed(2)}%</td>
      <td><strong>${(m.oob_score * 100).toFixed(2)}%</strong></td>
      <td><strong>${(m.test_accuracy * 100).toFixed(2)}%</strong></td>
      <td>${(m.test_f1_macro * 100).toFixed(2)}%</td>
      <td>${(m.test_precision_macro * 100).toFixed(2)}%</td>
      <td>${(m.test_recall_macro * 100).toFixed(2)}%</td>
      <td>${(m.test_roc_auc_ovr * 100).toFixed(2)}%</td>
      <td>
        ${isChamp ? '<span class="table-badge green">Optimized 🏆</span>' : '<span class="table-badge yellow">Baseline</span>'}
      </td>
    `;
    tbody.appendChild(tr);
  });
}
