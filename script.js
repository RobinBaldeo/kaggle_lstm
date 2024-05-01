// Set dimensions and margins
const width = 800, height = 600;
const margin = { top: 20, right: 20, bottom: 20, left: 20 };

// Create the SVG container
const svg = d3.select("#graph")
    .attr("width", width)
    .attr("height", height);

// Append a container to hold the graph elements
const container = svg.append("g");

// Define color scale for nodes
const color = d3.scaleOrdinal(d3.schemeCategory10);

// Define the zoom behavior
const zoom = d3.zoom()
    .scaleExtent([0.5, 4])
    .on("zoom", event => {
        container.attr("transform", event.transform);
    });

// Apply the zoom behavior to the SVG
svg.call(zoom);

let simulation, link, node, label;

// Load the JSON data
d3.json("data2.json").then(data => {
    // Initialize the simulation
    simulation = d3.forceSimulation(data.nodes)
        .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
        .force("charge", d3.forceManyBody().strength(-300))
        .force("center", d3.forceCenter(width / 2, height / 2));

    // Create elements for links, nodes, and labels
    link = container.append("g").attr("class", "links").selectAll("line");
    node = container.append("g").attr("class", "nodes").selectAll("circle");
    label = container.append("g").attr("class", "labels").selectAll("text");

    // Populate classification filter dropdown
    const classificationFilter = d3.select("#classificationFilter");

    // Set the default radio button to "dep"
    d3.select('input[name="radioOption"][value="dep"]').property("checked", true);

    // Load the selection for the multi-select box when "dep" is selected
    const classifications = new Set(data.links.map(link => link.value[1]));
    classifications.forEach(type => {
        classificationFilter.append("option").text(type).attr("value", type);
    });

    // Handle radio button change
    d3.selectAll('input[name="radioOption"]').on("change", function() {
        if (this.value === "dep") {
            // Clear the multi-select box and populate it with classifications
            classificationFilter.html("");
            classifications.forEach(type => {
                classificationFilter.append("option").text(type).attr("value", type);
            });
        } else if (this.value === "ind") {
            // Clear the multi-select box and populate it with top 10 sources
            classificationFilter.html("");
            const sourceFrequency = data.links.reduce((freq, link) => {
                freq[link.source.id] = (freq[link.source.id] || 0) + 1;
                return freq;
            }, {});
            const topSources = Object.keys(sourceFrequency)
                .sort((a, b) => sourceFrequency[b] - sourceFrequency[a])
                .slice(0, 10);
            topSources.forEach(source => {
                classificationFilter.append("option").text(source).attr("value", source);
            });
        }
        updateGraph();
    });

    // Handle multi-select box change
    classificationFilter.on("change", function() {
        updateGraph();
    });

    // Initial graph rendering
    updateGraph();

    // Function to update the graph based on the selected classification and radio button value
    function updateGraph() {
        const radioValue = d3.select('input[name="radioOption"]:checked').node();
        let filteredLinks = data.links;
        let filteredNodes = data.nodes;

        if (radioValue && radioValue.value === "dep") {
            const selectedClassifications = classificationFilter.property("value");
            filteredLinks = data.links.filter(link => selectedClassifications.includes(link.value[1]));
            const filteredNodeIds = new Set(filteredLinks.flatMap(l => [l.source.id, l.target.id]));
            filteredNodes = data.nodes.filter(node => filteredNodeIds.has(node.id));
        } else if (radioValue && radioValue.value === "ind") {
            const selectedSources = classificationFilter.property("value");
            filteredLinks = data.links.filter(link => selectedSources.includes(link.source.id));
            const filteredNodeIds = new Set(filteredLinks.flatMap(l => [l.source.id, l.target.id]));
            filteredNodes = data.nodes.filter(node => filteredNodeIds.has(node.id));
        }

        link = link.data(filteredLinks, d => `${d.source.id}-${d.target.id}`).join("line")
            .attr("stroke", "gray").attr("stroke-width", d => Math.sqrt(d.value[0]));
        node = node.data(filteredNodes, d => d.id).join("circle")
            .attr("r", 10).attr("fill", d => color(d.group[0])).call(drag(simulation));
        label = label.data(filteredNodes, d => d.id).join("text")
            .text(d => d.id).attr("dx", 12).attr("dy", ".35em");

        simulation.nodes(filteredNodes);
        simulation.force("link").links(filteredLinks);
        simulation.alpha(1).restart();
    }

    // Drag functionality for the nodes
    function drag(simulation) {
        return d3.drag()
            .on("start", event => {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            })
            .on("drag", event => {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            })
            .on("end", event => {
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            });
    }

    simulation.on("tick", () => {
        link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
        node.attr("cx", d => d.x).attr("cy", d => d.y);
        label.attr("x", d => d.x).attr("y", d => d.y + 4);
    });
});