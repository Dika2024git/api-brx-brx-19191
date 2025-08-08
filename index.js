// index.js

/**
 * AI Chatbot Express.js
 * Dibuat untuk memenuhi spesifikasi kompleks dengan konfigurasi berbasis XML.
 * Fitur: Intent Detection, Entity Recognition, Context Management, API Fallback,
 * Multi-language, Fuzzy Search, Relevance Scoring, dan lainnya.
 */

const express = require('express');
const fs = require('fs').promises;
const { parseStringPromise } = require('xml2js');
const Fuse = require('fuse.js');
const { WordTokenizer } = require('natural');
const LanguageDetect = require('language-detect');
const axios = require('axios');

const app = express();

// Variabel global untuk menyimpan konfigurasi dan data dari XML
let chatbotConfig = null;
// Variabel global untuk menyimpan instance Fuse.js yang sudah diinisialisasi
let fuseInstances = {
    intents: null,
    qa: {} // Akan diisi per bahasa, misal: qa: { id: FuseInstance, en: FuseInstance }
};
// Penyimpanan memori percakapan (dalam produksi, gunakan database seperti Redis)
const conversationMemory = {};

/**
 * Membersihkan objek hasil parse dari xml2js (menghilangkan array tunggal).
 * @param {object} obj Objek yang akan dibersihkan.
 * @returns {object} Objek yang sudah bersih.
 */
function cleanXmlObject(obj) {
    const newObj = {};
    for (const k in obj) {
        if (obj[k] && obj[k].length === 1 && typeof obj[k][0] === 'object' && Object.keys(obj[k][0]).length > 0) {
            if (obj[k][0]._) {
                 newObj[k] = obj[k][0]._;
            } else if (obj[k][0].$){
                 newObj[k] = { ...obj[k][0].$, value: obj[k][0]._ };
            } else {
                 newObj[k] = cleanXmlObject(obj[k][0]);
            }
        } else if (Array.isArray(obj[k])) {
            newObj[k] = obj[k].map(item => {
                if (typeof item === 'object' && item !== null && !Array.isArray(item)) {
                    if(item._) return { ...item.$, value: item._ };
                    return cleanXmlObject(item);
                }
                return item;
            });
        } else {
            newObj[k] = obj[k];
        }
    }
    return newObj;
}


/**
 * Memuat dan mem-parsing data.xml ke dalam variabel chatbotConfig.
 * Juga menginisialisasi instance Fuse.js untuk pencarian.
 */
async function loadAndPrepareData() {
    try {
        console.log("Membaca data.xml...");
        const xmlData = await fs.readFile('data.xml', 'utf8');
        const parsedJs = await parseStringPromise(xmlData, { explicitArray: true, mergeAttrs: true });
        
        // Membersihkan objek dari struktur array yang tidak perlu dari xml2js
        const root = parsedJs.chatbotConfig;
        chatbotConfig = {
            settings: cleanXmlObject(root.settings[0]),
            intents: root.intents[0].intent.map(i => cleanXmlObject({intent: [i]})).map(i => i.intent),
            entities: root.entities[0].entity.map(e => cleanXmlObject({entity: [e]})).map(e => e.entity),
            qaItems: root.qaItems[0].item.map(item => cleanXmlObject({item: [item]})).map(item => item.item),
            contexts: root.contexts ? (root.contexts[0].context || []).map(c => cleanXmlObject({context: [c]})).map(c => c.context) : [],
        };

        console.log("Data berhasil dimuat dan diproses.");

        // Inisialisasi Fuse.js untuk Intent Detection
        const intentOptions = {
            keys: ['keywords.k.value'],
            includeScore: true,
            threshold: parseFloat(chatbotConfig.settings.defaultThreshold) || 0.4,
            useExtendedSearch: true,
        };
        fuseInstances.intents = new Fuse(chatbotConfig.intents, intentOptions);

        // Inisialisasi Fuse.js untuk Q&A per bahasa
        const allQaItems = [...chatbotConfig.qaItems, ...chatbotConfig.contexts.flatMap(c => c.item || [])];
        const languages = [...new Set(allQaItems.map(item => item.lang))];
        
        for (const lang of languages) {
            const langQaItems = allQaItems.filter(item => item.lang === lang);
            const qaOptions = {
                keys: ['questions.q.value'],
                includeScore: true,
                includeMatches: true,
                threshold: parseFloat(chatbotConfig.settings.defaultThreshold) || 0.4,
            };
            fuseInstances.qa[lang] = new Fuse(langQaItems, qaOptions);
        }

        console.log("Inisialisasi Fuse.js selesai untuk intents dan Q&A.");

    } catch (error) {
        console.error("Gagal memuat atau memproses data.xml:", error);
        process.exit(1); // Keluar jika konfigurasi gagal dimuat
    }
}

/**
 * Mendeteksi bahasa dari teks input.
 * @param {string} text - Teks input dari pengguna.
 * @returns {string} Kode bahasa (misal: 'id', 'en').
 */
function detectLanguage(text) {
    if (chatbotConfig.settings.autoDetectLanguage !== 'true') return 'id'; // Default jika dinonaktifkan
    const detector = new LanguageDetect();
    const result = detector.detect(text, 1);
    // Mapping dari nama bahasa ke kode
    if (result.length > 0) {
        if (result[0][0].toLowerCase() === 'indonesian') return 'id';
        if (result[0][0].toLowerCase() === 'english') return 'en';
    }
    return 'id'; // Fallback ke bahasa Indonesia
}

/**
 * Mendapatkan jawaban acak dari sebuah item Q&A.
 * @param {object} item - Item Q&A yang cocok.
 * @returns {string} Jawaban acak.
 */
function getRandomAnswer(item) {
    const answers = item.answers.a;
    if (Array.isArray(answers)) {
        const randomIndex = Math.floor(Math.random() * answers.length);
        return answers[randomIndex].value || answers[randomIndex];
    }
    return answers.value || answers;
}

/**
 * Mengenali entitas dalam teks berdasarkan data.xml.
 * @param {string[]} tokens - Array token dari input pengguna.
 * @returns {object} Objek berisi entitas yang dikenali.
 */
function recognizeEntities(tokens) {
    const foundEntities = {};
    chatbotConfig.entities.forEach(entity => {
        const entityValues = Array.isArray(entity.values.v) ? entity.values.v.map(v => v.toLowerCase()) : [entity.values.v.toLowerCase()];
        tokens.forEach(token => {
            if (entityValues.includes(token.toLowerCase())) {
                foundEntities[entity.name] = token;
            }
        });
    });
    return foundEntities;
}

/**
 * Mengganti placeholder entitas di jawaban dengan nilai yang ditemukan.
 * @param {string} answer - Template jawaban.
 * @param {object} entities - Objek entitas yang ditemukan.
 * @returns {string} Jawaban yang sudah dipersonalisasi.
 */
function personalizeAnswer(answer, entities) {
    let personalized = answer;
    for (const key in entities) {
        personalized = personalized.replace(`{${key}}`, entities[key]);
    }
    return personalized;
}


// Endpoint utama chatbot
app.get('/chat', async (req, res) => {
    const userInput = req.query.q;
    const sessionId = req.query.sessionId;

    if (!userInput || !sessionId) {
        return res.status(400).json({ error: "Parameter 'q' (pertanyaan) dan 'sessionId' harus ada." });
    }

    // Inisialisasi memori jika sesi baru
    if (!conversationMemory[sessionId]) {
        conversationMemory[sessionId] = {
            history: [],
            context: null, // Konteks percakapan saat ini
        };
    }
    const session = conversationMemory[sessionId];

    try {
        // 1. Deteksi Bahasa
        const lang = detectLanguage(userInput);

        // 2. Tokenisasi
        const tokenizerType = chatbotConfig.settings.tokenizer || 'WordTokenizer';
        const tokenizer = new (require('natural'))[tokenizerType]();
        const tokens = tokenizer.tokenize(userInput.toLowerCase());
        
        // 3. Entity Recognition
        const foundEntities = recognizeEntities(tokens);

        let bestMatch = null;
        let responseSource = "local_fallback";

        // 4. Contextual Search
        if (session.context) {
            const contextData = chatbotConfig.contexts.find(c => c.id === session.context);
            if (contextData && contextData.item) {
                const contextFuse = new Fuse(Array.isArray(contextData.item) ? contextData.item : [contextData.item], { keys: ['questions.q.value'], includeScore: true, threshold: 0.5 });
                const contextResult = contextFuse.search(userInput);
                if (contextResult.length > 0) {
                    bestMatch = contextResult[0];
                }
            }
        }
        
        // 5. Jika tidak ada di konteks, cari di Q&A umum
        if (!bestMatch) {
            // 5a. Intent Detection
            const intentResults = fuseInstances.intents.search(userInput);
            let detectedIntent = 'unknown';
            if (intentResults.length > 0) {
                detectedIntent = intentResults[0].item.name;
            }

            // 5b. Q&A Search berdasarkan intent dan bahasa
            const qaFuse = fuseInstances.qa[lang];
            if (qaFuse) {
                const searchResults = qaFuse.search(userInput);
                
                // Filter hasil berdasarkan intent yang terdeteksi untuk meningkatkan relevansi
                const relevantResults = searchResults.filter(r => r.item.intent === detectedIntent);
                
                if (relevantResults.length > 0) {
                    // Terapkan pembobotan (relevance scoring)
                    relevantResults.sort((a, b) => {
                        const weightA = parseFloat(a.item.relevanceWeight || 1.0);
                        const weightB = parseFloat(b.item.relevanceWeight || 1.0);
                        // Skor lebih rendah lebih baik di Fuse.js
                        return (a.score / weightA) - (b.score / weightB);
                    });
                    bestMatch = relevantResults[0];
                } else if (searchResults.length > 0) {
                    // Jika tidak ada yang cocok dengan intent, ambil hasil terbaik dari pencarian umum
                    bestMatch = searchResults[0];
                }
            }
        }

        // 6. Proses Jawaban atau Fallback
        if (bestMatch && bestMatch.score < (parseFloat(bestMatch.item.threshold) || parseFloat(chatbotConfig.settings.defaultThreshold))) {
            let answer = getRandomAnswer(bestMatch.item);
            answer = personalizeAnswer(answer, foundEntities);
            
            // Update konteks untuk jawaban selanjutnya
            session.context = bestMatch.item.nextContextId || null;
            responseSource = "local_qa";

            session.history.push({ user: userInput, bot: answer });
            return res.json({
                answer: answer,
                intent: bestMatch.item.intent,
                score: bestMatch.score,
                source: responseSource,
                context: session.context,
                language: lang,
                entities: foundEntities
            });
        } else {
            // 7. Fallback ke API Eksternal
            const fallbackApi = chatbotConfig.settings.apiFallback;
            if (fallbackApi && fallbackApi.url) {
                try {
                    console.log(`Tidak ditemukan jawaban lokal, fallback ke API: ${fallbackApi.url}`);
                    const apiResponse = await axios.get(fallbackApi.url, { params: { q: userInput } });
                    
                    // Asumsi API mengembalikan format { answer: "..." }
                    const apiAnswer = apiResponse.data.answer || "Maaf, API eksternal juga bingung.";
                    session.history.push({ user: userInput, bot: apiAnswer });
                    session.context = null; // Reset konteks setelah fallback

                    return res.json({
                        answer: apiAnswer,
                        intent: "api_fallback",
                        source: "api_fallback",
                        context: null,
                        language: lang
                    });
                } catch (apiError) {
                    console.error("Error saat menghubungi API fallback:", apiError.message);
                }
            }

            // 8. Jawaban Fallback Terakhir dari XML
            const fallbackItem = chatbotConfig.qaItems.find(item => item.intent === 'fallback' && item.lang === lang);
            if (fallbackItem) {
                const fallbackAnswer = getRandomAnswer(fallbackItem);
                session.history.push({ user: userInput, bot: fallbackAnswer });
                session.context = null;
                return res.json({
                    answer: fallbackAnswer,
                    intent: "fallback",
                    source: "local_fallback",
                    context: null,
                    language: lang
                });
            }
        }

    } catch (error) {
        console.error("Terjadi error di endpoint /chat:", error);
        res.status(500).json({ error: "Waduh, ada yang error di server nih. Coba lagi nanti ya." });
    }
});

// Jalankan server setelah data siap
loadAndPrepareData();

module.exports = app;
