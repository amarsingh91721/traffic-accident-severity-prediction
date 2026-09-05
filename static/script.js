let map, markers = [], severityChart, roadChart;

// 0) populate year dropdown on load
document.addEventListener('DOMContentLoaded', async () => {
  const sel = document.getElementById('year');
  sel.innerHTML = '<option value="">All</option>';
  try {
    const years = await fetch('/years').then(r => r.json());
    years.forEach(y => {
      const o = document.createElement('option');
      o.value = y;
      o.textContent = y;
      sel.append(o);
    });
  } catch (err) {
    console.error('Failed to load years:', err);
  }
});

// 1) Google Maps init
window.initMap = function() {
  map = new google.maps.Map(document.getElementById('map'), {
    center: { lat: 37.0902, lng: -95.7129 }, zoom: 4
  });
};

// 2) clear markers
function clearMarkers() {
  markers.forEach(m => m.setMap(null));
  markers = [];
}

// 3) fetch & plot hotspots
async function loadHotspots() {
  clearMarkers();
  const year = document.getElementById('year').value;
  const wc   = document.getElementById('weather_code').value;
  const rc   = document.getElementById('road_class_code').value;

  const pts = (await fetch(
    `/hotspots?year=${year}&weather_code=${wc}&road_class_code=${rc}`
  ).then(r => r.json()))
    .filter(p => p.lat && p.lng);

  if (!pts.length) { alert('No hotspots found.'); return; }

  const bounds = new google.maps.LatLngBounds();
  pts.forEach(p => {
    const marker = new google.maps.Marker({
      position: { lat: p.lat, lng: p.lng }, map,
      icon: { path: google.maps.SymbolPath.CIRCLE, scale:6, fillColor:'red', fillOpacity:0.7, strokeWeight:0 }
    });
    marker.addListener('click', () => showDetails(p.lat, p.lng));
    markers.push(marker);
    bounds.extend(marker.getPosition());
  });
  map.fitBounds(bounds);
}

// 4) on‑click: details + charts + prediction + conditions
async function showDetails(lat, lng) {
  const wc = document.getElementById('weather_code').value;
  const rc = document.getElementById('road_class_code').value;

  // Accident details
  const resp = await fetch(`/details?lat=${lat}&lng=${lng}`).then(r => r.json());
  document.getElementById('accidentCount').textContent =
    `Total accidents: ${resp.total_accidents}`;

  // Severity chart
  const sevCtx = document.getElementById('severityChart').getContext('2d');
  if (severityChart) severityChart.destroy();
  severityChart = new Chart(sevCtx, {
    type: 'bar',
    data: {
      labels: Object.keys(resp.casualty_by_severity),
      datasets: [{
        data: Object.values(resp.casualty_by_severity),
        backgroundColor: ['#4caf50','#ff9800','#f44336']
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend:{ display:false } },
      scales: { y:{ beginAtZero:true, title:{ display:true, text:'# casualties' } } }
    }
  });

  // Road surface chart
  const roadCtx = document.getElementById('roadChart').getContext('2d');
  if (roadChart) roadChart.destroy();
  roadChart = new Chart(roadCtx, {
    type: 'pie',
    data: {
      labels: Object.keys(resp.road_surface),
      datasets: [{ data: Object.values(resp.road_surface) }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend:{ position:'bottom' } }
    }
  });

  // Model prediction + conditions
  const hour = new Date().getHours();
  const predResp = await fetch(
    `/predict?` +
      `lat=${lat}&lng=${lng}` +
      `&weather_code=${wc}` +
      `&road_class_code=${rc}` +
      `&hour=${hour}`
  ).then(r => r.json());

  const predEl = document.getElementById('modelPrediction');
  const condEl = document.getElementById('conditions');

  if (predResp.error) {
    predEl.textContent = 'Prediction: error';
    condEl.textContent = '';
  } else {
    predEl.textContent = `Prediction: ${predResp.prediction}`;
    const c = predResp.conditions;
    condEl.textContent =
      `Weather: ${c.weather} · Road: ${c.road_class} · Surface: ${c.road_surface} · Lighting: ${c.lighting} · Hour: ${c.hour}`;
  }
}
