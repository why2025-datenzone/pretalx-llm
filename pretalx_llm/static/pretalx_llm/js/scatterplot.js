const globalData = document.getElementById("global-data");
let searchUrl = globalData.dataset.url;

fetch(searchUrl)
.then(
    response => response.json(), 
    _reason => {
        const plotElement = document.querySelector("#plot");
        plotElement.innerHTML = "Failed to fetch data, please reload the page!";
    })
.then(data => {
    const submissions = data.submissions || [];

    if (submissions.length == 0) {
        const plotElement = document.querySelector("#plot");
        plotElement.innerHTML = "No data to display";
        return
    }
    const series = submissions.map(item => ({
        x: item.x,
        y: item.y,
        title: item.title,
        highlight: item.highlight,
        url: item.url
    }));

    const options = {
        chart: {
            type: 'scatter',
            height: 600,
            zoom: {
                type: 'xy',
                enabled: true
            },
            toolbar: {
                show: true
            },
            events: {
                markerClick(event, chartContext, opts) {
                    console.log(opts)
                    location.href = opts.w.config.series[opts.seriesIndex].data[opts.dataPointIndex].url
                }
            }
        },
        series: [{
            name: 'Highlighted Submissions',
            data: series.filter(item => item.highlight).map((item, index) => ({ ...item, index, url: item.url }))
        },
        {
            name: 'Regular Submissions',
            data: series.filter(item => !item.highlight).map((item, index) => ({ ...item, index, url: item.url }))
        }],
        xaxis: {
            type: 'numeric',
            labels: {
                formatter: (val) => val.toFixed(1)
            },
            tooltip: {
                enabled: false
            },
            axisBorder: {
                show: false
            },
            axisTicks: {
                show: false
            },
            crosshairs: {
                show: false
            }
        },
        yaxis: {
            type: 'numeric',
            labels: {
                formatter: (val) => val.toFixed(1)
            }
        },
        tooltip: {
            enabled: true,
            custom: ({ seriesIndex, dataPointIndex, w }) => {
                const point = seriesIndex === 0
                    ? series.filter(item => item.highlight)[dataPointIndex]
                    : series.filter(item => !item.highlight)[dataPointIndex];
                return `<div style="padding: 8px;">${point.title}</div>`;
            }
        },
        grid: {
            xaxis: {
                lines: {
                    show: false
                }
            },
            yaxis: {
                lines: {
                    show: false
                }
            }
        },
        markers: {
            size: 6,
            colors: ["#FF4560", "#008FFB"], // Highlighted: red, Regular: blue
            hover: {
                sizeOffset: 4
            },
        }
    };

    const plotElement = document.querySelector("#plot");
    plotElement.innerHTML = ""; // Clear existing content

    const chart = new ApexCharts(plotElement, options);
    chart.render();
});
