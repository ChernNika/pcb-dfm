(() => {
    const byId = id => document.getElementById(id);
    const all = selector => document.querySelectorAll(selector);
    const cards = Array.from(all('.card'));
    const severities = ['err', 'warn', 'info', 'nc'];
    const filters = {classes: new Set(), severities: new Set(severities), code: null, layer: '', query: ''};

    function matches(data) {
        return (!filters.classes.size || filters.classes.has(data.cls))
            && filters.severities.has(data.sev)
            && (!filters.code || data.code === filters.code)
            && (!filters.layer || data.layer === filters.layer)
            && (!filters.query || data.text.includes(filters.query));
    }

    function applyFilters() {
        const visible = new Set();
        for (const card of cards) {
            card.hidden = !matches(card.dataset);
            if (!card.hidden) visible.add(card.id.slice(1));
        }
        const code = filters.code ? ' · дефект ' + filters.code : '';
        byId('shown').textContent = 'показано ' + visible.size + ' из ' + cards.length + code;
        for (const marker of all('.overview a')) {
            marker.querySelector('circle').classList.toggle('hide', !visible.has(marker.dataset.i));
        }
        byId('nothing').hidden = visible.size > 0;
    }

    for (const chip of all('.chip[data-sev], .chip[data-cls]')) {
        chip.onclick = () => {
            const value = chip.dataset.sev || chip.dataset.cls;
            const active = chip.dataset.sev ? filters.severities : filters.classes;
            if (active.has(value)) active.delete(value);
            else active.add(value);
            chip.classList.toggle('on', active.has(value));
            applyFilters();
        };
    }

    byId('layer').onchange = event => {
        filters.layer = event.target.value;
        applyFilters();
    };
    byId('q').oninput = event => {
        filters.query = event.target.value.trim().toLowerCase();
        applyFilters();
    };

    const sortKeys = {
        sev: card => -Number(card.dataset.rank),
        prob: card => -Number(card.dataset.prob),
        code: card => card.dataset.cls + card.dataset.code,
    };
    byId('sort').onchange = event => {
        const key = sortKeys[event.target.value];
        const sorted = [...cards].sort((a, b) => {
            const first = key(a);
            const second = key(b);
            return first < second ? -1 : first > second ? 1 : 0;
        });
        for (const card of sorted) byId('list').appendChild(card);
    };

    for (const button of all('button[data-code]')) {
        button.onclick = () => {
            const {code, sev} = button.dataset;
            filters.code = filters.code === code && !sev ? null : code;
            if (sev) {
                filters.severities = new Set([sev]);
                for (const chip of all('.chip[data-sev]')) chip.classList.toggle('on', chip.dataset.sev === sev);
            }
            applyFilters();
            document.querySelector('.controls').scrollIntoView({behavior: 'smooth'});
        };
    }

    byId('reset').onclick = () => {
        filters.classes.clear();
        filters.severities = new Set(severities);
        filters.code = null;
        filters.layer = '';
        filters.query = '';
        byId('q').value = '';
        byId('layer').value = '';
        for (const chip of all('.chip[data-sev]')) chip.classList.add('on');
        for (const chip of all('.chip[data-cls]')) chip.classList.remove('on');
        applyFilters();
    };

    function flashCard(id) {
        const card = byId(id);
        if (!card) return;
        card.hidden = false;
        card.classList.add('flash');
        setTimeout(() => card.classList.remove('flash'), 1600);
    }
    for (const marker of all('.overview a')) {
        marker.addEventListener('click', () => flashCard('f' + marker.dataset.i));
    }
    if (location.hash) flashCard(location.hash.slice(1));

    // В локальном файле доступны все фильтры. Выделение объектов требует связи с плагином.
    if (location.protocol.startsWith('http')) {
        document.body.classList.add('live');
        for (const button of all('button.kicad')) {
            button.onclick = async () => {
                const feedback = button.nextElementSibling;
                feedback.textContent = '…';
                try {
                    const query = 'select?uids=' + encodeURIComponent(button.dataset.uids)
                        + '&refs=' + encodeURIComponent(button.dataset.refs);
                    const response = await fetch(query);
                    const result = await response.json();
                    feedback.textContent = result.ok ? 'выделено объектов: ' + result.selected : 'KiCad: ' + result.error;
                } catch {
                    feedback.textContent = 'нет связи с плагином (KiCad закрыт или прошло 30 мин)';
                }
            };
        }
    }

    const themes = ['', 'light', 'dark'];
    const themeLabels = {'': 'тема: авто', light: 'светлая', dark: 'тёмная'};
    let theme = '';
    try {
        theme = localStorage.getItem('dfm-theme') || '';
    } catch {
        // Браузер может запретить хранилище для файла, открытого с диска.
    }
    function setTheme(value) {
        theme = value;
        if (value) document.documentElement.dataset.theme = value;
        else delete document.documentElement.dataset.theme;
        byId('theme').textContent = '◐ ' + themeLabels[value];
        try { localStorage.setItem('dfm-theme', value); } catch {}
    }
    setTheme(theme);
    byId('theme').onclick = () => setTheme(themes[(themes.indexOf(theme) + 1) % themes.length]);
    applyFilters();
})();
