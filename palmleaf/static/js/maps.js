let mapInstance = null;

function initMap(discoveryLoc, repositoryLoc) {
    if (mapInstance) mapInstance.remove();

    const lat = discoveryLoc ? discoveryLoc.latitude : 10.7828;
    const lng = discoveryLoc ? discoveryLoc.longitude : 79.1318;

    mapInstance = L.map('map').setView([lat, lng], 8);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    }).addTo(mapInstance);

    if (discoveryLoc) {
        L.circleMarker([discoveryLoc.latitude, discoveryLoc.longitude], {
            color: '#8b2500', fillColor: '#c68642', fillOpacity: 0.9, radius: 9
        }).addTo(mapInstance)
         .bindPopup(`<b>🏛️ Ancient Discovery Location</b><br>${discoveryLoc.name || 'Excavation Site'}`)
         .openPopup();
    }

    if (repositoryLoc) {
        L.marker([repositoryLoc.latitude, repositoryLoc.longitude])
         .addTo(mapInstance)
         .bindPopup(`<b>📚 Modern Repository</b><br>${repositoryLoc.name || 'Preservation Archive'}`);
    }
}
