function renderMindMap(containerId, rootData) {
    const container = document.getElementById(containerId);
    d3.select(`#${containerId}`).selectAll("*").remove();

    const width = container.clientWidth || 800;
    const height = 420;

    const COLOR = {
        root:    "#8b2500",
        dynasty: "#c68642",
        subject: "#4a6b82",
        child:   "#5a7a5a",
        script:  "#7a4a82"
    };

    const ICON = {
        root:    "📜",
        dynasty: "🏛️",
        subject: "📚",
        script:  "✍️",
        child:   "🔹"
    };

    const svg = d3.select(`#${containerId}`)
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .style("cursor", "grab");

    // Zoom & Pan
    const g = svg.append("g");
    svg.call(d3.zoom()
        .scaleExtent([0.4, 3])
        .on("zoom", (event) => g.attr("transform", event.transform))
    );

    // Arrow marker
    svg.append("defs").append("marker")
        .attr("id", "arrow")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 28).attr("refY", 0)
        .attr("markerWidth", 6).attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .attr("d", "M0,-5L10,0L0,5")
        .attr("fill", "#c68642")
        .attr("opacity", 0.6);

    const simulation = d3.forceSimulation(rootData.nodes)
        .force("link", d3.forceLink(rootData.links).id(d => d.id).distance(150))
        .force("charge", d3.forceManyBody().strength(-500))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide(40));

    // Curved links
    const link = g.append("g").selectAll("path")
        .data(rootData.links).enter().append("path")
        .attr("fill", "none")
        .attr("stroke", "#c68642")
        .attr("stroke-width", 1.8)
        .attr("stroke-opacity", 0.5)
        .attr("marker-end", "url(#arrow)");

    // Node groups
    const node = g.append("g").selectAll("g")
        .data(rootData.nodes).enter().append("g")
        .style("cursor", "pointer")
        .call(d3.drag()
            .on("start", (event, d) => {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x; d.fy = d.y;
                svg.style("cursor", "grabbing");
            })
            .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
            .on("end", (event, d) => {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null; d.fy = null;
                svg.style("cursor", "grab");
            })
        );

    // Glow filter
    const defs = svg.select("defs");
    const filter = defs.append("filter").attr("id", "glow");
    filter.append("feGaussianBlur").attr("stdDeviation", "3").attr("result", "coloredBlur");
    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "coloredBlur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Circle with glow on root
    node.append("circle")
        .attr("r", d => d.type === "root" ? 28 : 20)
        .attr("fill", d => COLOR[d.type] || COLOR.child)
        .attr("stroke", "#fff")
        .attr("stroke-width", 2)
        .attr("filter", d => d.type === "root" ? "url(#glow)" : null)
        .attr("opacity", 0)
        .transition().duration(600).delay((d, i) => i * 80)
        .attr("opacity", 1);

    // Emoji icon inside node
    node.append("text")
        .text(d => ICON[d.type] || ICON.child)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "central")
        .attr("font-size", d => d.type === "root" ? "16px" : "12px")
        .attr("y", 1)
        .attr("opacity", 0)
        .transition().duration(600).delay((d, i) => i * 80)
        .attr("opacity", 1);

    // Label below node
    node.append("text")
        .text(d => d.label.length > 20 ? d.label.slice(0, 18) + "…" : d.label)
        .attr("text-anchor", "middle")
        .attr("y", d => d.type === "root" ? 42 : 34)
        .attr("font-size", d => d.type === "root" ? "13px" : "11px")
        .attr("font-weight", d => d.type === "root" ? "700" : "500")
        .attr("fill", "#2d241e")
        .attr("opacity", 0)
        .transition().duration(600).delay((d, i) => i * 80)
        .attr("opacity", 1);

    // Tooltip
    const tooltip = d3.select("body").append("div")
        .style("position", "absolute")
        .style("background", "rgba(0,0,0,0.8)")
        .style("color", "#fff")
        .style("padding", "6px 12px")
        .style("border-radius", "6px")
        .style("font-size", "12px")
        .style("pointer-events", "none")
        .style("opacity", 0)
        .style("z-index", 9999);

    node.on("mouseover", (event, d) => {
        tooltip.transition().duration(150).style("opacity", 1);
        tooltip.html(`<strong>${d.label}</strong><br/><span style="color:#c68642">${d.type}</span>`)
            .style("left", (event.pageX + 12) + "px")
            .style("top", (event.pageY - 10) + "px");
    }).on("mousemove", (event) => {
        tooltip.style("left", (event.pageX + 12) + "px").style("top", (event.pageY - 10) + "px");
    }).on("mouseout", () => {
        tooltip.transition().duration(200).style("opacity", 0);
    });

    simulation.on("tick", () => {
        link.attr("d", d => {
            const dx = d.target.x - d.source.x;
            const dy = d.target.y - d.source.y;
            const dr = Math.sqrt(dx * dx + dy * dy) * 1.2;
            return `M${d.source.x},${d.source.y}A${dr},${dr} 0 0,1 ${d.target.x},${d.target.y}`;
        });
        node.attr("transform", d => `translate(${d.x},${d.y})`);
    });

    // Cleanup tooltip on re-render
    return () => tooltip.remove();
}
