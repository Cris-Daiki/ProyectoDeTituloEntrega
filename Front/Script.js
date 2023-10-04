
console.log("ME GUAUG")
src="https://cdn.jsdelivr.net/npm/chart.js"

src="https://cdn.jsdelivr.net/npm/chartjs-plugin-chart-financial@1.1.0/dist/chartjs-plugin-chart-financial.min.js"
const ctx = document.getElementById('myChart');
ctx.style.width = '100%';
ctx.style.height = '100%';
let chartType = document.getElementById('chart-type').value;

let chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  scales: {
    y: {
      beginAtZero: true
    }
  }
};

let chartData = {
  labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange','Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
  datasets: [{
    label: 'Valor Accion',
    data: [12, 19, 3, 5, 2, 3,2,3,4,5,6,7,8,9,6,5,4,3,2,4,5,6,7],
    backgroundColor: ['black'],
    borderColor: 'rgba(255, 99, 132, 1)',
    borderWidth: 2
  }]
};

let myChart = new Chart(ctx, {
  type: chartType,
  data: chartData,
  options: chartOptions
});

document.getElementById('chart-type').addEventListener('change', function() {
  chartType = this.value;
  myChart.destroy();
  myChart = new Chart(ctx, {
    type: chartType,
    data: chartData,
    options: chartOptions
  });
});