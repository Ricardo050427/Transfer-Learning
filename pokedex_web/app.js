/**
 * =============================================================================
 * POKÉDEX AI — SISTEMA DE INFERENCIA Y CONTROL EN EL NAVEGADOR
 * TensorFlow.js + Web Audio API (Síntesis 8-bit) + MediaDevices (Cámara)
 * =============================================================================
 */

(function () {
  'use strict';

  // ===========================================================================
  // 1. ESTADO DE LA APLICACIÓN Y CONSTANTES
  // ===========================================================================
  const STATE = {
    model: null,
    classNames: [],
    isModelLoading: true,
    isCameraActive: false,
    currentFacingMode: 'environment', // 'environment' (trasera) o 'user' (frontal)
    mediaStream: null,
    isScanning: false,
    audioEnabled: true,
    audioCtx: null,
    currentSampleIndex: 0,
    samplesList: ['pikachu', 'charizard', 'bulbasaur', 'blastoise', 'gengar', 'mewtwo'],
    isTflite: false
  };

  // Mapeo de IDs nacionales para sprites oficiales de la 1ª Generación
  const POKEMON_DEX_MAP = {
    'Bulbasaur': 1, 'Ivysaur': 2, 'Venusaur': 3, 'Charmander': 4, 'Charmeleon': 5,
    'Charizard': 6, 'Squirtle': 7, 'Wartortle': 8, 'Blastoise': 9, 'Caterpie': 10,
    'Metapod': 11, 'Butterfree': 12, 'Weedle': 13, 'Kakuna': 14, 'Beedrill': 15,
    'Pidgey': 16, 'Pidgeotto': 17, 'Pidgeot': 18, 'Rattata': 19, 'Raticate': 20,
    'Spearow': 21, 'Fearow': 22, 'Ekans': 23, 'Arbok': 24, 'Pikachu': 25,
    'Raichu': 26, 'Sandshrew': 27, 'Sandslash': 28, 'Nidoran♀': 29, 'Nidorina': 30,
    'Nidoqueen': 31, 'Nidoran♂': 32, 'Nidorino': 33, 'Nidoking': 34, 'Clefairy': 35,
    'Clefable': 36, 'Vulpix': 37, 'Ninetales': 38, 'Jigglypuff': 39, 'Wigglytuff': 40,
    'Zubat': 41, 'Golbat': 42, 'Oddish': 43, 'Gloom': 44, 'Vileplume': 45,
    'Paras': 46, 'Parasect': 47, 'Venonat': 48, 'Venomoth': 49, 'Diglett': 50,
    'Dugtrio': 51, 'Meowth': 52, 'Persian': 53, 'Psyduck': 54, 'Golduck': 55,
    'Mankey': 56, 'Primeape': 57, 'Growlithe': 58, 'Arcanine': 59, 'Poliwag': 60,
    'Poliwhirl': 61, 'Poliwrath': 62, 'Abra': 63, 'Kadabra': 64, 'Alakazam': 65,
    'Machop': 66, 'Machoke': 67, 'Machamp': 68, 'Bellsprout': 69, 'Weepinbell': 70,
    'Victreebel': 71, 'Tentacool': 72, 'Tentacruel': 73, 'Geodude': 74, 'Graveler': 75,
    'Golem': 76, 'Ponyta': 77, 'Rapidash': 78, 'Slowpoke': 79, 'Slowbro': 80,
    'Magnemite': 81, 'Magneton': 82, 'Farfetchd': 83, 'Doduo': 84, 'Dodrio': 85,
    'Seel': 86, 'Dewgong': 87, 'Grimer': 88, 'Muk': 89, 'Shellder': 90,
    'Cloyster': 91, 'Gastly': 92, 'Haunter': 93, 'Gengar': 94, 'Onix': 95,
    'Drowzee': 96, 'Hypno': 97, 'Krabby': 98, 'Kingler': 99, 'Voltorb': 100,
    'Electrode': 101, 'Exeggcute': 102, 'Exeggutor': 103, 'Cubone': 104, 'Marowak': 105,
    'Hitmonlee': 106, 'Hitmonchan': 107, 'Lickitung': 108, 'Koffing': 109, 'Weezing': 110,
    'Rhyhorn': 111, 'Rhydon': 112, 'Chansey': 113, 'Tangela': 114, 'Kangaskhan': 115,
    'Horsea': 116, 'Seadra': 117, 'Goldeen': 118, 'Seaking': 119, 'Staryu': 120,
    'Starmie': 121, 'MrMime': 122, 'Scyther': 123, 'Jynx': 124, 'Electabuzz': 125,
    'Magmar': 126, 'Pinsir': 127, 'Tauros': 128, 'Magikarp': 129, 'Gyarados': 130,
    'Lapras': 131, 'Ditto': 132, 'Eevee': 133, 'Vaporeon': 134, 'Jolteon': 135,
    'Flareon': 136, 'Porygon': 137, 'Omanyte': 138, 'Omastar': 139, 'Kabuto': 140,
    'Kabutops': 141, 'Aerodactyl': 142, 'Snorlax': 143, 'Articuno': 144, 'Zapdos': 145,
    'Moltres': 146, 'Dratini': 147, 'Dragonair': 148, 'Dragonite': 149, 'Mewtwo': 150,
    'Mew': 151
  };

  // ===========================================================================
  // 2. ELEMENTOS DEL DOM
  // ===========================================================================
  const DOM = {
    sensorGem: document.getElementById('sensorGem'),
    ledRed: document.getElementById('ledRed'),
    ledYellow: document.getElementById('ledYellow'),
    ledGreen: document.getElementById('ledGreen'),
    soundToggle: document.getElementById('soundToggle'),
    soundIcon: document.getElementById('soundIcon'),
    screenViewport: document.getElementById('screenViewport'),
    laserScanner: document.getElementById('laserScanner'),
    cameraFeed: document.getElementById('cameraFeed'),
    imagePreview: document.getElementById('imagePreview'),
    idleState: document.getElementById('idleState'),
    screenStatus: document.getElementById('screenStatus'),
    hudMode: document.getElementById('hudMode'),
    hudFps: document.getElementById('hudFps'),
    btnToggleCamera: document.getElementById('btnToggleCamera'),
    btnSwitchCamera: document.getElementById('btnSwitchCamera'),
    btnClear: document.getElementById('btnClear'),
    fileInput: document.getElementById('fileInput'),
    btnPaste: document.getElementById('btnPaste'),
    dexId: document.getElementById('dexId'),
    dexName: document.getElementById('dexName'),
    pokeSprite: document.getElementById('pokeSprite'),
    spritePlaceholder: document.getElementById('spritePlaceholder'),
    confidenceValue: document.getElementById('confidenceValue'),
    confidenceBar: document.getElementById('confidenceBar'),
    predictionList: document.getElementById('predictionList'),
    statusLog: document.getElementById('statusLog'),
    statusIndicatorBar: document.getElementById('statusIndicatorBar'),
    btnSnap: document.getElementById('btnSnap'),
    btnHelp: document.getElementById('btnHelp'),
    infoModal: document.getElementById('infoModal'),
    modalClose: document.getElementById('modalClose'),
    dpadUp: document.getElementById('dpadUp'),
    dpadDown: document.getElementById('dpadDown'),
    dpadLeft: document.getElementById('dpadLeft'),
    dpadRight: document.getElementById('dpadRight')
  };

  // ===========================================================================
  // 3. SINTETIZADOR DE AUDIO 8-BIT (WEB AUDIO API)
  // ===========================================================================
  function getAudioContext() {
    if (!STATE.audioCtx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        STATE.audioCtx = new AudioCtx();
      }
    }
    if (STATE.audioCtx && STATE.audioCtx.state === 'suspended') {
      STATE.audioCtx.resume();
    }
    return STATE.audioCtx;
  }

  function playTone(freq, type = 'square', duration = 0.08, volume = 0.15) {
    if (!STATE.audioEnabled) return;
    try {
      const ctx = getAudioContext();
      if (!ctx) return;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, ctx.currentTime);
      gain.gain.setValueAtTime(volume, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + duration);
    } catch (e) {
      // Audio seguro
    }
  }

  // Sonidos Retro Específicos
  const SOUNDS = {
    boot: () => {
      // Arpegio Game Boy
      [523.25, 659.25, 783.99, 1046.50].forEach((freq, i) => {
        setTimeout(() => playTone(freq, 'square', 0.12, 0.2), i * 90);
      });
    },
    click: () => playTone(880, 'square', 0.04, 0.1),
    keypad: (num) => playTone(300 + num * 60, 'triangle', 0.05, 0.15),
    scan: () => {
      if (!STATE.audioEnabled) return;
      try {
        const ctx = getAudioContext();
        if (!ctx) return;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(220, ctx.currentTime);
        osc.frequency.linearRampToValueAtTime(880, ctx.currentTime + 0.3);
        gain.gain.setValueAtTime(0.12, ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.01, ctx.currentTime + 0.3);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start();
        osc.stop(ctx.currentTime + 0.3);
      } catch (e) {}
    },
    success: () => {
      // Fanfarria clásica de detección
      const notes = [659.25, 659.25, 659.25, 523.25, 659.25, 783.99];
      const times = [0, 100, 200, 320, 420, 560];
      notes.forEach((f, i) => {
        setTimeout(() => playTone(f, 'square', 0.1, 0.18), times[i]);
      });
    },
    error: () => {
      playTone(180, 'sawtooth', 0.2, 0.2);
    }
  };

  // ===========================================================================
  // 4. CARGA DEL MODELO Y LISTA DE CLASES
  // ===========================================================================
  async function initPokedex() {
    updateStatus('CONECTANDO...', 'led-yellow');
    
    try {
      // 1. Cargar lista de clases
      const classesResp = await fetch('./model/class_names.json');
      if (!classesResp.ok) throw new Error('No se pudo cargar class_names.json');
      STATE.classNames = await classesResp.json();
      console.log(`[Pokédex] Clases cargadas: ${STATE.classNames.length}`);

      // 2. Cargar modelo
      updateStatus(`CARGANDO RED (${STATE.classNames.length} CLASES)...`, 'led-yellow');
      
      let modelLoaded = false;
      if (window.tflite) {
        try {
          console.log('[Pokédex] Intentando cargar modelo TFLite...');
          tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/');
          STATE.model = await tflite.loadTFLiteModel('./model/model.tflite');
          STATE.isTflite = true;
          modelLoaded = true;
          console.log('[Pokédex] ¡Modelo TFLite WASM cargado exitosamente!');
        } catch (tfliteErr) {
          console.warn('[Pokédex] TFLite aviso:', tfliteErr);
        }
      }

      if (!modelLoaded) {
        try {
          STATE.model = await tf.loadLayersModel('./model/model.json');
          STATE.isTflite = false;
          modelLoaded = true;
          console.log('[Pokédex] Modelo LayersModel cargado correctamente.');
        } catch (layersErr) {
          console.warn('[Pokédex] tf.loadLayersModel falló:', layersErr);
          throw layersErr;
        }
      }

      // Warmup del modelo
      try {
        const dummy = tf.zeros([1, 224, 224, 3]);
        const out = STATE.model.predict(dummy);
        if (out instanceof tf.Tensor) out.dispose();
        else if (out && typeof out === 'object') Object.values(out).forEach(t => t && t.dispose && t.dispose());
        dummy.dispose();
      } catch (warmupErr) {
        console.warn('[Pokédex] Warmup aviso:', warmupErr);
      }

      STATE.isModelLoading = false;
      updateStatus('POKÉDEX V2.0 ONLINE', 'led-green');
      DOM.statusIndicatorBar.textContent = `${STATE.classNames.length} POKÉMON LISTOS`;
      DOM.hudMode.textContent = 'MODO: ESPERA';
      SOUNDS.boot();

    } catch (err) {
      console.error('[Pokédex Error]', err);
      updateStatus('ERROR AL CARGAR IA', 'led-red');
      DOM.statusLog.innerHTML = `<span style="color:#FF3B53">ERROR: ${err.message}</span>`;
      SOUNDS.error();
    }
  }

  function updateStatus(text, ledClass) {
    DOM.statusLog.innerHTML = `<span class="log-cursor">█</span> ${text}`;
    DOM.screenStatus.textContent = text;
    DOM.ledRed.classList.remove('active');
    DOM.ledYellow.classList.remove('active');
    DOM.ledGreen.classList.remove('active');
    
    if (ledClass === 'led-green') DOM.ledGreen.classList.add('active');
    else if (ledClass === 'led-yellow') DOM.ledYellow.classList.add('active');
    else if (ledClass === 'led-red') DOM.ledRed.classList.add('active');
  }

  // ===========================================================================
  // 5. CÁMARA EN VIVO
  // ===========================================================================
  async function toggleCamera() {
    SOUNDS.click();
    if (STATE.isCameraActive) {
      stopCamera();
    } else {
      await startCamera();
    }
  }

  async function startCamera() {
    try {
      updateStatus('INICIANDO LENTE...', 'led-yellow');
      
      const constraints = {
        video: {
          facingMode: { ideal: STATE.currentFacingMode },
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      };

      STATE.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      DOM.cameraFeed.srcObject = STATE.mediaStream;
      
      await new Promise(resolve => {
        DOM.cameraFeed.onloadedmetadata = () => {
          DOM.cameraFeed.play();
          resolve();
        };
      });

      DOM.cameraFeed.classList.remove('hidden');
      DOM.imagePreview.classList.add('hidden');
      DOM.idleState.classList.add('hidden');
      DOM.laserScanner.classList.add('active');
      DOM.sensorGem.classList.add('scanning');

      STATE.isCameraActive = true;
      DOM.btnToggleCamera.querySelector('.btn-pixel-text').textContent = 'DETENER';
      DOM.hudMode.textContent = 'MODO: CÁMARA';
      updateStatus('ESCÁNER ACTIVO', 'led-green');

      // Bucle de clasificación continua (1 cuadro por cada 400ms para fluidez y bajo consumo)
      loopCameraInference();

    } catch (err) {
      console.error('Error de cámara:', err);
      updateStatus('CÁMARA NO DISPONIBLE', 'led-red');
      SOUNDS.error();
    }
  }

  function stopCamera() {
    if (STATE.mediaStream) {
      STATE.mediaStream.getTracks().forEach(track => track.stop());
      STATE.mediaStream = null;
    }
    DOM.cameraFeed.classList.add('hidden');
    DOM.laserScanner.classList.remove('active');
    DOM.sensorGem.classList.remove('scanning');
    STATE.isCameraActive = false;
    DOM.btnToggleCamera.querySelector('.btn-pixel-text').textContent = 'CÁMARA';
    DOM.hudMode.textContent = 'MODO: ESPERA';
    
    if (DOM.imagePreview.classList.contains('hidden')) {
      DOM.idleState.classList.remove('hidden');
    }
    updateStatus('CÁMARA DETENIDA', 'led-green');
  }

  async function switchCamera() {
    SOUNDS.click();
    if (!STATE.isCameraActive) return;
    STATE.currentFacingMode = STATE.currentFacingMode === 'environment' ? 'user' : 'environment';
    stopCamera();
    await startCamera();
  }

  let cameraLoopTimer = null;
  async function loopCameraInference() {
    if (!STATE.isCameraActive) return;
    if (!STATE.isScanning && STATE.model && DOM.cameraFeed.readyState === 4) {
      await classifySource(DOM.cameraFeed, false);
    }
    cameraLoopTimer = setTimeout(loopCameraInference, 450);
  }

  // ===========================================================================
  // 6. PIPELINE DE CLASIFICACIÓN CON TENSORFLOW.JS
  // ===========================================================================
  async function classifySource(mediaElement, triggerEffects = true) {
    if (!STATE.model || STATE.classNames.length === 0) return;
    if (STATE.isScanning) return;

    STATE.isScanning = true;
    if (triggerEffects) {
      DOM.laserScanner.classList.add('active');
      DOM.sensorGem.classList.add('scanning');
      SOUNDS.scan();
    }

    try {
      const startTime = performance.now();

      // Tensor preprocessing: el modelo ya incluye Rescaling(1./127.5, offset=-1)
      // por lo que alimentamos píxeles crudos [0, 255] redimensionados a 224x224
      let probabilities = [];
      if (STATE.isTflite) {
        const rawTensor = tf.browser.fromPixels(mediaElement);
        const resized = tf.image.resizeBilinear(rawTensor, [224, 224]);
        const batched = resized.toFloat().expandDims(0); // [1, 224, 224, 3]
        const output = STATE.model.predict(batched);
        const outTensor = (output instanceof tf.Tensor) ? output : Object.values(output)[0];
        const rawData = await outTensor.data();
        probabilities = Array.from(rawData);

        // Liberación de tensores en memoria
        rawTensor.dispose();
        resized.dispose();
        batched.dispose();
        if (output instanceof tf.Tensor) output.dispose();
        else if (output && typeof output === 'object') Object.values(output).forEach(t => t && t.dispose && t.dispose());
      } else {
        probabilities = tf.tidy(() => {
          const rawTensor = tf.browser.fromPixels(mediaElement);
          const resized = tf.image.resizeBilinear(rawTensor, [224, 224]);
          const batched = resized.toFloat().expandDims(0); // [1, 224, 224, 3]
          const preds = STATE.model.predict(batched);
          return Array.from(preds.dataSync());
        });
      }

      const inferenceTime = (performance.now() - startTime).toFixed(0);
      DOM.hudFps.textContent = `${inferenceTime}ms`;

      // Ordenar las clases de mayor a menor probabilidad
      const indices = Array.from(probabilities.keys())
        .sort((a, b) => probabilities[b] - probabilities[a]);

      const top1Idx = indices[0];
      const top1Prob = probabilities[top1Idx];
      const top1Name = STATE.classNames[top1Idx];

      // Formatear las top-3 para la pantalla secundaria
      const topResults = indices.slice(0, 3).map(idx => ({
        name: STATE.classNames[idx],
        percent: (probabilities[idx] * 100).toFixed(1)
      }));

      // Renderizar resultados en pantalla
      displayPrediction(top1Name, top1Prob, topResults);

      if (triggerEffects && top1Prob > 0.15) {
        SOUNDS.success();
      }

    } catch (err) {
      console.error('Error durante inferencia:', err);
      updateStatus('ERROR EN INFERENCIA', 'led-red');
    } finally {
      STATE.isScanning = false;
      if (triggerEffects) {
        DOM.laserScanner.classList.remove('active');
        DOM.sensorGem.classList.remove('scanning');
      }
    }
  }

  // ===========================================================================
  // 7. RENDERIZADO DE RESULTADOS EN LA PANTALLA SECUNDARIA
  // ===========================================================================
  function displayPrediction(speciesName, confidence, topList) {
    const cleanName = speciesName.replace('Mega', ' (Mega)');
    DOM.dexName.textContent = cleanName.toUpperCase();
    
    const percent = (confidence * 100).toFixed(1);
    DOM.confidenceValue.textContent = `${percent}%`;
    DOM.confidenceBar.style.width = `${Math.min(100, Math.max(5, percent))}%`;

    // Resolver Número de Pokédex y Sprite Oficial
    const baseName = speciesName.split('Mega')[0].trim();
    const dexNum = POKEMON_DEX_MAP[baseName] || POKEMON_DEX_MAP[speciesName];
    
    if (dexNum) {
      DOM.dexId.textContent = `N.° ${String(dexNum).padStart(3, '0')}`;
      
      // Sprite pixel art oficial de PokeAPI (Gen 1 / Red-Blue o animación)
      const spriteUrl = `https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/${dexNum}.png`;
      DOM.pokeSprite.src = spriteUrl;
      DOM.pokeSprite.classList.remove('hidden');
      DOM.spritePlaceholder.classList.add('hidden');
    } else {
      DOM.dexId.textContent = 'N.° ???';
      DOM.pokeSprite.classList.add('hidden');
      DOM.spritePlaceholder.classList.remove('hidden');
    }

    // Renderizar lista top 3
    DOM.predictionList.innerHTML = topList.map((item, idx) => `
      <li class="pred-item">
        <span>${idx + 1}. ${item.name}</span>
        <span>${item.percent}%</span>
      </li>
    `).join('');

    updateStatus(`ID: ${cleanName.toUpperCase()}`, 'led-green');
  }

  // ===========================================================================
  // 8. CARGA DE IMÁGENES (SUBIDA / MUESTRAS RÁPIDAS)
  // ===========================================================================
  function loadImageFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    
    SOUNDS.click();
    stopCamera();

    const reader = new FileReader();
    reader.onload = (e) => {
      DOM.imagePreview.src = e.target.result;
      DOM.imagePreview.onload = () => {
        DOM.imagePreview.classList.remove('hidden');
        DOM.idleState.classList.add('hidden');
        DOM.hudMode.textContent = 'MODO: FOTO';
        classifySource(DOM.imagePreview, true);
      };
    };
    reader.readAsDataURL(file);
  }

  function loadSamplePokemon(sampleName) {
    SOUNDS.click();
    stopCamera();

    const sampleUrl = `./samples/${sampleName}.jpg`;
    DOM.imagePreview.src = sampleUrl;
    DOM.imagePreview.onload = () => {
      DOM.imagePreview.classList.remove('hidden');
      DOM.idleState.classList.add('hidden');
      DOM.hudMode.textContent = 'MODO: MUESTRA';
      classifySource(DOM.imagePreview, true);
    };
  }

  function loadImageFromUrl(url) {
    SOUNDS.click();
    stopCamera();
    updateStatus('CARGANDO URL...', 'led-yellow');

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      DOM.imagePreview.src = url;
      DOM.imagePreview.classList.remove('hidden');
      DOM.idleState.classList.add('hidden');
      DOM.hudMode.textContent = 'MODO: PEGAR';
      classifySource(DOM.imagePreview, true);
    };
    img.onerror = () => {
      updateStatus('ERROR AL CARGAR URL', 'led-red');
      SOUNDS.error();
    };
    img.src = url;
  }

  async function pasteImageFromClipboard() {
    SOUNDS.click();
    try {
      if (navigator.clipboard && navigator.clipboard.read) {
        const clipboardItems = await navigator.clipboard.read();
        for (const item of clipboardItems) {
          const imageType = item.types.find(t => t.startsWith('image/'));
          if (imageType) {
            const blob = await item.getType(imageType);
            loadImageFile(blob);
            updateStatus('IMAGEN PEGADA', 'led-green');
            return;
          }
        }
      }
    } catch (err) {
      console.warn('[Pokédex Clipboard]', err);
    }
    
    // Fallback o mensaje informativo si se requiere Ctrl+V
    updateStatus('PRESIONA CTRL + V', 'led-yellow');
  }

  function clearAll() {
    SOUNDS.click();
    stopCamera();
    DOM.imagePreview.classList.add('hidden');
    DOM.imagePreview.src = '';
    DOM.idleState.classList.remove('hidden');
    
    DOM.dexId.textContent = 'N.° ---';
    DOM.dexName.textContent = 'POKÉMON DESCONOCIDO';
    DOM.confidenceValue.textContent = '0.0%';
    DOM.confidenceBar.style.width = '0%';
    DOM.pokeSprite.classList.add('hidden');
    DOM.spritePlaceholder.classList.remove('hidden');
    
    DOM.predictionList.innerHTML = `
      <li class="pred-item empty"><span>1. ---</span><span>--%</span></li>
      <li class="pred-item empty"><span>2. ---</span><span>--%</span></li>
      <li class="pred-item empty"><span>3. ---</span><span>--%</span></li>
    `;
    updateStatus('MEMORIA REINICIADA', 'led-green');
  }

  // ===========================================================================
  // 9. EVENT LISTENERS
  // ===========================================================================
  // Cámara
  DOM.btnToggleCamera.addEventListener('click', toggleCamera);
  DOM.btnSwitchCamera.addEventListener('click', switchCamera);
  DOM.btnClear.addEventListener('click', clearAll);

  // Subir imagen de archivo
  DOM.fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      loadImageFile(e.target.files[0]);
    }
  });

  // Botón Pegar desde portapapeles
  if (DOM.btnPaste) {
    DOM.btnPaste.addEventListener('click', pasteImageFromClipboard);
  }

  // Pegar imagen directamente desde el portapapeles (Ctrl + V global)
  window.addEventListener('paste', (e) => {
    if (!e.clipboardData) return;

    // 1. Revisar si hay un archivo o imagen binaria (ej. "Copiar imagen" en Google)
    const items = e.clipboardData.items;
    if (items) {
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
          const blob = items[i].getAsFile();
          if (blob) {
            e.preventDefault();
            SOUNDS.click();
            loadImageFile(blob);
            updateStatus('IMAGEN PEGADA (CTRL+V)', 'led-green');
            return;
          }
        }
      }
    }

    // 2. Revisar si es texto con URL de imagen copiada (ej. "Copiar dirección de imagen")
    const text = e.clipboardData.getData('text');
    if (text && (text.startsWith('http://') || text.startsWith('https://') || text.startsWith('data:image/'))) {
      e.preventDefault();
      loadImageFromUrl(text.trim());
    }
  });

  // Drag and Drop en la pantalla del escáner
  DOM.screenViewport.addEventListener('dragover', (e) => {
    e.preventDefault();
    DOM.screenViewport.style.borderColor = '#00FF66';
  });

  DOM.screenViewport.addEventListener('dragleave', () => {
    DOM.screenViewport.style.borderColor = 'var(--pokedex-border)';
  });

  DOM.screenViewport.addEventListener('drop', (e) => {
    e.preventDefault();
    DOM.screenViewport.style.borderColor = 'var(--pokedex-border)';
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      loadImageFile(e.dataTransfer.files[0]);
    }
  });

  // Muestras rápidas
  document.querySelectorAll('.sample-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const sample = chip.getAttribute('data-sample');
      loadSamplePokemon(sample);
    });
  });

  // Teclado numérico (efecto de sonido retro)
  document.querySelectorAll('.key-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = parseInt(btn.getAttribute('data-key'), 10);
      SOUNDS.keypad(key);
    });
  });

  // Botón SCAN (Dispara escaneo del fotograma actual)
  DOM.btnSnap.addEventListener('click', () => {
    SOUNDS.click();
    if (STATE.isCameraActive) {
      classifySource(DOM.cameraFeed, true);
    } else if (!DOM.imagePreview.classList.contains('hidden')) {
      classifySource(DOM.imagePreview, true);
    } else {
      // Si no hay imagen, cargar la primera muestra
      loadSamplePokemon(STATE.samplesList[0]);
    }
  });

  // Control D-Pad
  DOM.dpadLeft.addEventListener('click', () => {
    SOUNDS.click();
    STATE.currentSampleIndex = (STATE.currentSampleIndex - 1 + STATE.samplesList.length) % STATE.samplesList.length;
    loadSamplePokemon(STATE.samplesList[STATE.currentSampleIndex]);
  });

  DOM.dpadRight.addEventListener('click', () => {
    SOUNDS.click();
    STATE.currentSampleIndex = (STATE.currentSampleIndex + 1) % STATE.samplesList.length;
    loadSamplePokemon(STATE.samplesList[STATE.currentSampleIndex]);
  });

  DOM.dpadUp.addEventListener('click', () => DOM.btnSnap.click());
  DOM.dpadDown.addEventListener('click', () => DOM.btnSnap.click());

  // Mute / Unmute
  DOM.soundToggle.addEventListener('click', () => {
    STATE.audioEnabled = !STATE.audioEnabled;
    DOM.soundIcon.textContent = STATE.audioEnabled ? '🔊' : '🔇';
    DOM.soundToggle.title = STATE.audioEnabled ? 'Sonido 8-bit Activado' : 'Sonido Silenciado';
    if (STATE.audioEnabled) SOUNDS.click();
  });

  // Modal INFO
  DOM.btnHelp.addEventListener('click', () => {
    SOUNDS.click();
    DOM.infoModal.classList.remove('hidden');
  });

  DOM.modalClose.addEventListener('click', () => {
    SOUNDS.click();
    DOM.infoModal.classList.add('hidden');
  });

  DOM.infoModal.addEventListener('click', (e) => {
    if (e.target === DOM.infoModal) {
      DOM.infoModal.classList.add('hidden');
    }
  });

  // Inicializar al cargar el DOM
  window.addEventListener('DOMContentLoaded', initPokedex);

})();
