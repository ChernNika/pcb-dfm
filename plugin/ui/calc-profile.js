/* Общий выбор профиля. Изменения на странице DFM действуют только в текущем сеансе. */
async function initCalculatorProfiles(apply) {
    const select = document.getElementById('calc-profile');
    const hint = document.getElementById('profile-note');
    if (location.protocol === 'file:') return;
    try {
        const options = await (await fetch('options')).json();
        if (options.error) throw new Error(options.error);
        const profiles = options.profiles.filter(p => !p.error && p.parameters);
        const manual = select.value;
        let context = {};
        try { context = JSON.parse(sessionStorage.getItem('dfm-calc-profile')) || {}; } catch {}
        for (const item of profiles) select.add(new Option(item.name, item.path));
        const path = context.path ?? options.last?.profile ?? '';
        select.value = profiles.some(p => p.path === path) ? path : manual;
        function useProfile() {
            const p = profiles.find(p => p.path === select.value);
            const fields = {};
            if (p) {
                for (const [key, field] of Object.entries(p.parameters.fields)) fields[key] = field.value;
                if (context.path === p.path) Object.assign(fields, context.changes);
            }
            apply(p?.parameters, fields);
            hint.textContent = p ? 'Значения профиля можно изменить для этого расчёта.' : '';
        }
        select.onchange = () => {
            if (context.path !== select.value) context = {path:select.value, changes:{}};
            try { sessionStorage.setItem('dfm-calc-profile', JSON.stringify(context)); } catch {}
            useProfile();
        };
        useProfile();
    } catch {
        hint.textContent = 'Профили недоступны. Можно задать параметры вручную.';
    }
}
