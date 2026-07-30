const form = document.getElementById('predictionForm');
  const tenure = document.getElementById('tenure');
  const tenureRange = document.getElementById('tenureRange');
  tenureRange.addEventListener('input', () => tenure.value = tenureRange.value);
  tenure.addEventListener('input', () => tenureRange.value = Math.max(0, Math.min(72, Number(tenure.value || 0))));

  function getPayload() {
    const data = new FormData(form);
    return {
      SeniorCitizen: Number(data.get('SeniorCitizen')),
      Partner: data.get('Partner'),
      Dependents: data.get('Dependents'),
      tenure: Number(data.get('tenure')),
      OnlineSecurity: data.get('OnlineSecurity'),
      OnlineBackup: data.get('OnlineBackup'),
      DeviceProtection: data.get('DeviceProtection'),
      TechSupport: data.get('TechSupport'),
      Contract: data.get('Contract'),
      PaperlessBilling: data.get('PaperlessBilling'),
      PaymentMethod: data.get('PaymentMethod'),
      MonthlyCharges: Number(data.get('MonthlyCharges')),
      TotalCharges: Number(data.get('TotalCharges'))
    };
  }

  // UI fallback only. Real production risk must come from the notebook's saved model API.
  function localEstimate(p) {
    let score = 24;
    if (p.Contract === 'Month-to-month') score += 22;
    if (p.tenure <= 6) score += 19; else if (p.tenure >= 36) score -= 14;
    if (p.MonthlyCharges >= 70) score += 13;
    if (p.PaymentMethod === 'Electronic check') score += 9;
    if (p.PaperlessBilling === 'Yes') score += 4;
    if (p.OnlineSecurity === 'No') score += 7;
    if (p.TechSupport === 'No') score += 8;
    if (p.OnlineBackup === 'No') score += 4;
    if (p.DeviceProtection === 'No') score += 4;
    if (p.Contract === 'Two year') score -= 20;
    return Math.max(4, Math.min(96, score));
  }

  function render(score, p, source) {
    const rounded = Math.round(score);
    const riskValue = document.getElementById('riskValue');
    const label = document.getElementById('riskLabel');
    const fill = document.getElementById('gaugeFill');
    const factorList = document.getElementById('factorList');
    riskValue.textContent = rounded + '%';
    const high = rounded >= 65, medium = rounded >= 35 && rounded < 65;
    label.textContent = high ? 'High churn risk' : medium ? 'Moderate churn risk' : 'Low churn risk';
    const color = high ? '#ff7b7b' : medium ? '#fbbf24' : '#5eead4';
    fill.style.borderTopColor = color; fill.style.borderRightColor = color;
    fill.style.transform = `rotate(${45 + (rounded / 100) * 180}deg)`;

    const factors = [];
    if (p.Contract === 'Month-to-month') factors.push(['Month-to-month contract', 'Higher risk', false]);
    else factors.push([p.Contract + ' contract', 'Protective', true]);
    if (p.tenure <= 6) factors.push(['First six months', 'Critical period', false]);
    else if (p.tenure >= 36) factors.push(['Long tenure', 'Protective', true]);
    if (p.MonthlyCharges >= 70) factors.push(['Charges above $70', 'Higher risk', false]);
    if (p.TechSupport === 'No') factors.push(['No tech support', 'Higher risk', false]);
    if (p.OnlineSecurity === 'Yes') factors.push(['Online security active', 'Protective', true]);
    factorList.innerHTML = factors.slice(0, 4).map(([name, tag, good]) => `<div class="factor"><span>${name}</span><span class="tag ${good ? 'good' : ''}">${tag}</span></div>`).join('') || '<div class="factor"><span>Balanced customer profile</span><span class="tag good">Stable</span></div>';

    document.getElementById('actionTitle').textContent = high ? 'Priority retention outreach' : medium ? 'Proactive service check-in' : 'Maintain customer experience';
    document.getElementById('actionText').textContent = high
      ? 'Offer a support-led retention package, review monthly cost, and discuss a longer contract option.'
      : medium ? 'Confirm service satisfaction and promote online security, backup, device protection, or technical support where relevant.'
      : 'Continue reliable service and use a loyalty offer near the next contract milestone.';
    document.getElementById('apiStatus').innerHTML = source === 'api' ? 'Prediction source: <strong>live model API</strong>.' : 'Prediction source: <strong>local interface demo</strong>. Connect the saved model through <code>/predict</code> for production results.';
    const toast = document.getElementById('toast'); toast.classList.add('show'); setTimeout(() => toast.classList.remove('show'), 1900);
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const payload = getPayload();
    try {
      const response = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (!response.ok) throw new Error('API unavailable');
      const result = await response.json();
      const probability = result.churn_probability ?? result.probability ?? (result.prediction === 1 ? 0.75 : 0.25);
      render(Number(probability) <= 1 ? Number(probability) * 100 : Number(probability), payload, 'api');
    } catch (error) {
      render(localEstimate(payload), payload, 'demo');
    }
  });

  form.addEventListener('reset', () => setTimeout(() => {
    tenureRange.value = 12;
    document.getElementById('riskValue').textContent = '--';
    document.getElementById('riskLabel').textContent = 'Enter customer details';
    document.getElementById('factorList').innerHTML = '<div class="factor"><span>No prediction yet</span><span class="tag good">Waiting</span></div>';
    document.getElementById('actionTitle').textContent = 'Retention action';
    document.getElementById('actionText').textContent = 'A recommended next step will appear after prediction.';
  }, 0));
