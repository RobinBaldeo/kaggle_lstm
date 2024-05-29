fetch('data1.json')
            .then(response => response.json())
            .then(data => {
                document.getElementById('nodeAlign').addEventListener('change', () => updateChart(data));
                document.getElementById('linkColor').addEventListener('change', () => updateChart(data));

                function updateChart(data) {
                    const nodeAlign = document.getElementById('nodeAlign').value;
                    const linkColor = document.getElementById('linkColor').value;

                    const width = 928;
                    const height = 600;
                    const format = d3.format(",.0f");

                    const svg = d3.select("#sankey")
                        .attr("width", width)
                        .attr("height", height)
                        .attr("viewBox", [0, 0, width, height])
                        .attr("style", "max-width: 100%; height: auto; font: 10px sans-serif;")
                        .html(""); // clear previous content

                    const sankey = d3.sankey()
                        .nodeId(d => d.id)
                        .nodeAlign(d3[nodeAlign])
                        .nodeWidth(15)
                        .nodePadding(10)
                        .extent([[1, 5], [width - 1, height - 5]]);

                    const { nodes, links } = sankey({
                        nodes: data.nodes.map(d => Object.assign({}, d)),
                        links: data.links.map(d => Object.assign({}, d))
                    });

                    const color = d3.scaleOrdinal(d3.schemeCategory10);

                    const rect = svg.append("g")
                        .attr("stroke", "#000")
                        .selectAll("rect")
                        .data(nodes)
                        .join("rect")
                        .attr("x", d => d.x0)
                        .attr("y", d => d.y0)
                        .attr("height", d => d.y1 - d.y0)
                        .attr("width", d => d.x1 - d.x0)
                        .attr("fill", d => color(d.category));

                    rect.append("title")
                        .text(d => `${d.id}\n${format(d.value)} TWh`);

                    const link = svg.append("g")
                        .attr("fill", "none")
                        .attr("stroke-opacity", 0.5)
                        .selectAll("g")
                        .data(links)
                        .join("g")
                        .style("mix-blend-mode", "multiply");

                    if (linkColor === "source-target") {
                        const gradient = link.append("linearGradient")
                            .attr("id", d => (d.uid = `link-${d.index}`))
                            .attr("gradientUnits", "userSpaceOnUse")
                            .attr("x1", d => d.source.x1)
                            .attr("x2", d => d.target.x0);
                        gradient.append("stop")
                            .attr("offset", "0%")
                            .attr("stop-color", d => color(d.source.category));
                        gradient.append("stop")
                            .attr("offset", "100%")
                            .attr("stop-color", d => color(d.target.category));
                    }

                    link.append("path")
                        .attr("d", d3.sankeyLinkHorizontal())
                        .attr("stroke", d => {
                            if (linkColor === "source-target") return `url(#link-${d.index})`;
                            if (linkColor === "source") return color(d.source.category);
                            if (linkColor === "target") return color(d.target.category);
                            return linkColor;
                        })
                        .attr("stroke-width", d => Math.max(1, d.width));

                    link.append("title")
                        .text(d => `${d.source.id} → ${d.target.id}\n${format(d.value)} TWh`);

                    svg.append("g")
                        .selectAll("text")
                        .data(nodes)
                        .join("text")
                        .attr("x", d => d.x0 < width / 2 ? d.x1 + 6 : d.x0 - 6)
                        .attr("y", d => (d.y1 + d.y0) / 2)
                        .attr("dy", "0.35em")
                        .attr("text-anchor", d => d.x0 < width / 2 ? "start" : "end")
                        .text(d => d.id);
                }

                updateChart(data);
            });