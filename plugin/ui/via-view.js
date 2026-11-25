const VIA_VIEW = `<section class="lab" id="pad">
  <div class="lab-head">
    <h3>Геометрия контактной площадки</h3>
    <p>Размеры ниже используются и для каплевидной площадки, и для графика брака. Перетащите отверстие на схеме, чтобы проверить выбранное смещение.</p>
  </div>
  <div class="grid">
    <div class="fig"><svg id="padSvg" role="img" aria-label="Контактная площадка, проводник, область допустимого смещения отверстия"></svg><div class="cap">мкм · перетащите отверстие</div></div>
    <div class="panel">
      <div class="controls">
        <div class="ctl"><label for="p_dvia">Диаметр КП D<sub>via</sub>, мкм</label><input required type="number" id="p_dvia" min="0.000001" step="any" value="400"></div>
        <div class="ctl"><label for="p_dd">Диаметр отверстия D<sub>drill</sub>, мкм</label><input required type="number" id="p_dd" min="0.000001" step="any" value="200"></div>
        <div class="ctl"><label for="p_w">Ширина проводника w<sub>tr_d</sub>, мкм</label><input required type="number" id="p_w" min="0.000001" step="any" value="150"></div>
        <div class="ctl"><label for="p_wmin">Мин. допустимая ширина w<sub>min</sub>, мкм</label><input required type="number" id="p_wmin" min="0.000001" step="any" value="150"></div>
        <div class="ctl"><label for="p_s3">Точность станка 3σ<sub>drill</sub>, мкм</label><input type="number" required id="p_s3" min="0" step="any" value="30"></div>
        <div class="ctl"><label for="b2">Допустимая дуга разрыва, класс 2, °</label><input type="number" required min="0" max="180" step="any" id="b2" value="90"></div>
        <div class="ctl"><label for="b1">Допустимая дуга разрыва, класс 1, °</label><input type="number" required min="0" max="180" step="any" id="b1" value="180"></div>
        <div class="ctl"><label for="p_ring">Минимальный поясок класса 3, мкм</label><input type="number" id="p_ring" value="50" min="0" step="any" required></div>
        <div class="ctl"><label>Класс применения</label><div class="seg" id="p_cls"><button type="button" data-v="3" aria-pressed="true">3</button><button type="button" data-v="2" aria-pressed="false">2</button><button type="button" data-v="1" aria-pressed="false">1</button></div></div>
        <div class="ctl"><label>Критерий годности</label><div class="seg" id="p_mode"><button type="button" data-v="author" aria-pressed="true">Поясок и ширина соединения</button><button type="button" data-v="ring" aria-pressed="false">Только поясок / разрыв</button></div></div>
      </div>
      <p id="via-error" role="status"></p>
      <div class="read" id="padRead"></div>
    </div>
  </div>
</section>


<section class="lab" id="tear">
  <div class="lab-head">
    <h3>Модель каплевидной КП и ширина перемычки</h3>
  </div>
  <div class="grid">
    <div class="fig"><svg id="tearSvg" role="img" aria-label="Каплевидная контактная площадка и перемычки"></svg><div class="cap">мкм · перетащите отверстие</div></div>
    <div class="panel">
      <div class="controls">
        <div class="ctl"><label>Класс применения</label><div class="seg" id="t_cls"><button type="button" data-v="3" aria-pressed="true">3</button><button type="button" data-v="2" aria-pressed="false">2</button><button type="button" data-v="1" aria-pressed="false">1</button></div></div>
        <div class="ctl"><label>Положение отверстия</label><button type="button" class="btn" id="t_reset">Вернуть в точку (0, R<sub>C</sub>)</button></div>
      </div>
      <div class="read" id="tearRead"></div>
    </div>
  </div>
</section>


<section class="lab" id="curves">
  <div class="lab-head">
    <h3>Доля брака КП в зависимости от точности станка</h3>
  </div>
  <div class="grid">
    <div class="fig"><svg id="curveSvg" viewBox="0 0 640 400" role="img" aria-label="Доля брака от точности станка"></svg></div>
    <div class="panel">
      <div class="controls">
        <div class="legend"><span><i style="border-color:var(--c3)"></i>класс 3</span><span><i style="border-color:var(--c2)"></i>класс 2</span><span><i style="border-color:var(--c1)"></i>класс 1</span><span><i class="d" style="border-color:var(--muted)"></i>без эллипса</span></div>
      </div>
      <div class="tablewrap"><table id="t5"></table></div>
      <label for="target">Целевая вероятность годности</label> <input id="target" type="number" value="0.997" min="0.000001" max="0.999999" step="any" required>
    </div>
  </div>
</section>`;
