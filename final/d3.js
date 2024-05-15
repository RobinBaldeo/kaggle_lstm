    // Set up dimensions and margins
    const margin = { top: 150, right: 150, bottom: 60, left: 100 };
    const width = 800 - margin.left - margin.right;
    const height = 250 - margin.top - margin.bottom;





    // Create SVG element
    const svg = d3.select("#slide-1")
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height * 3 + margin.top * 4 + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left- 20}, ${margin.top+ 20})`);



svg.append("text")
  .attr("transform", `translate(${(width+100) / 2}, ${-margin.top / 3})`)
  .attr("text-anchor", "middle")
  .attr("font-size", "24px")
  .attr("font-weight", "bold")
  .text("Final Solution Evolution Chart");


    // Define timeline data for each set
    const timelineData1 = [
      { date: 0, event: "LSTM Model 1", description: "Bi-Directional LSTM with CRF layer", value: -20 },
      { date: 3, event: "LSTM Model 2", description: "Bi-Directional LSTM with POS Embeddings and CRF layer", value: -20 },
      { date: 10, event: "LSTM Model 3", description: "Multiple Bi-Directional LSTM with POS Embeddings and CRF layer", value: -20 }
    ];

    const timelineData2 = [
      { date: 0, event: "Hybrid Model 4", description: "Bi-Directional LSTM with CRF layer and Tiny Bert Embeddings", value: -20 },
      { date: 5, event: "Hybrid Model 5", description: "Bi-Directional LSTM with CRF layer and Tiny Bert Embeddings and POS Enrichment", value: -20 },
      { date: 10, event: "Hybrid Model 6", description: "Multiple Bi-Directional LSTM with CRF layer and Tiny Bert Embeddings and POS Enrichment", value: -20 }
    ];

    const timelineData3 = [
      { date: 0, event: "Transformer Model 7", description: "Tiny Bert with stock head", value: -20 },
      { date: 5, event: "Transformer Model 8", description: "Small Deberta with stock head", value: -20 },
      { date: 10, event: "Transformer Model 9", description: "Small Deberta with custom head", value: -20 }
    ];


function wrapText(selection, width) {
  selection.each(function() {
    let text = d3.select(this),
        words = text.text().split(/\s+/).reverse(),
        word,
        line = [],
        lineNumber = 0,
        lineHeight = 1.1, // ems
        y = text.attr("y"),
        x = text.attr("x"),
        dy = parseFloat(text.attr("dy") || 0),
        tspan = text.text(null).append("tspan").attr("x", x).attr("y", y).attr("dy", dy + "em");

    while (word = words.pop()) {
      line.push(word);
      tspan.text(line.join(" "));
      if (line.length > 3) {
        line.pop();
        tspan.text(line.join(" "));
        line = [word];
        tspan = text.append("tspan").attr("x", x).attr("y", y).attr("dy", ++lineNumber * lineHeight + dy + "em").text(word);
      }
    }
  });
}


    // Create a function to generate the lollipop timeline chart for each set
function createLollipopChart(timelineData, yOffset) {
      // Set up scales
      const xScale = d3.scaleTime()
        .domain(d3.extent(timelineData, d => d.date))
        .range([0, width]);

      const yScale = d3.scaleLinear()
        .domain([-d3.max(timelineData, d => Math.abs(d.value)), d3.max(timelineData, d => Math.abs(d.value))])
        .range([height / 2, 0]);

      // Create x-axis without tick marks and labels
      const xAxis = d3.axisBottom(xScale)
        .tickFormat("")
        .tickSize(0);

      svg.append("g")
        .attr("class", "x-axis")
        .attr("transform", `translate(0, ${height / 2 + yOffset})`)
        .call(xAxis);

      // Create lollipop lines and circles
      svg.selectAll(`.lollipop-line-${yOffset}`)
        .data(timelineData)
        .enter()
        .append("line")

        .attr("class", "lollipop-line")
        .attr("x1", d => xScale(d.date))
        .attr("y1", height / 2 + yOffset)
        .attr("x2", d => xScale(d.date))
        .attr("y2", d => d.value >= 0 ? height / 2 + yOffset - yScale(d.value) : height / 2 + yOffset + yScale(d.value));

      svg.selectAll(`.lollipop-circle-${yOffset}`)
        .data(timelineData)
        .enter()
        .append("circle")
        .attr("class", "lollipop-circle")
        .attr("cx", d => xScale(d.date))
        .attr("cy", height / 2 + yOffset)
        .attr("r", 5);

      // Create event labels and descriptions
      const eventLabels = svg.selectAll(`.event-label-${yOffset}`)
        .data(timelineData)
        .enter()
        .append("text")
        .attr("class", `event-label-${yOffset}`)
        .attr("x", d => xScale(d.date) + 5)
        .attr("y", d => {
          const lineHeight = height / 2 + yOffset - yScale(d.value);
          return d.value >= 0 ? lineHeight - 20 : height / 2 + yOffset + yScale(d.value) + 15;
        })
        .attr("text-anchor", "right")
          .attr("font-weight", "bold")
        .text(d => d.event);

      const eventDescriptions = svg.selectAll(`.event-description-${yOffset}`)
        .data(timelineData)
        .enter()
        .append("text")

        .attr("class", `event-description-${yOffset}`)

        .attr("x", d => {
            console.log(xScale(d.date))
            return xScale(d.date)  + 5}
        )
        .attr("y", d => {
          const lineHeight = height / 2 + yOffset - yScale(d.value);
          return d.value >= 0 ? lineHeight - 5 : height / 2 + yOffset + yScale(d.value) + 35;
        })
        .attr("text-anchor", "left")
        .text(d => d.description)
        .call(wrapText, 150);




    }

    // Create the lollipop timeline charts for each set
    createLollipopChart(timelineData1, 0);
    createLollipopChart(timelineData2, height + margin.top);
    createLollipopChart(timelineData3, (height + margin.top) * 2);