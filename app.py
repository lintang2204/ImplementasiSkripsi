import hashlib
import io
import json
import pandas as pd
from PIL import Image
import streamlit as st
from model_registry import SCENARIOS, source_root
from inference import (PLANES, TISSUES, EXPECTED, THRESHOLD,
                       model_inventory, collect_images, predict_subject)

st.set_page_config(page_title='ASD–TC | MRI Classifier', page_icon='🧠', layout='wide')
st.markdown('''<style>
.block-container{max-width:1180px;padding-top:2.2rem}
[data-testid="stMetric"]{background:#f0f5fa;border:1px solid #dbe6f0;border-radius:12px;padding:16px}
</style>''', unsafe_allow_html=True)
st.caption('IMPLEMENTASI PENELITIAN · MRI STRUKTURAL')
st.title('Klasifikasi ASD dan TC')
st.write('Unggah 12 irisan satu subjek untuk menjalankan model EfficientNetB0 dan penggabungan probabilitas tiga bidang potong.')
with st.sidebar:
    st.header('Model penelitian')
    scenario = st.selectbox('Skenario', list(SCENARIOS), format_func=lambda s:SCENARIOS[s]['label'])
    inv = model_inventory(scenario)
    available = sum(r['available'] for r in inv)
    st.metric('Model tersedia', f'{available}/15')
    st.caption('5 fold × 3 bidang potong')
    st.write(f'Ambang akhir OOF: **{SCENARIOS[scenario]["threshold"]:.2f}**')
    st.caption('Model dibaca dari folder lokal; tidak diunggah ke layanan lain.')
    st.caption(str(source_root(scenario)))
    with st.expander('Kelengkapan model'):
        st.dataframe(pd.DataFrame(inv)[['plane','fold','available']], hide_index=True)

tab_input, tab_info = st.tabs(['Prediksi subjek', 'Metode & panduan'])
with tab_info:
    st.subheader('Masukan yang sesuai penelitian')
    st.write('Empat representasi ANAT, CSF, GM, WM pada axial, coronal, sagittal. Gunakan PNG 8-bit hasil slicing penelitian, bukan tangkapan layar atau MRI mentah.')
    st.code('SUBJEK_ANAT_axial_slice_80.png\nSUBJEK_CSF_axial_slice_80.png\n...\nSUBJEK_WM_sagittal_slice_90.png', language=None)
    st.write('Nama singkat seperti SUBJEK_ANAT_axial.png juga diterima. Awalan subjek harus sama pada semua berkas.')
    st.subheader('Urutan prediksi')
    st.write('Rata-rata 5 model per citra → rata-rata 4 representasi per bidang potong → rata-rata 3 bidang potong → keputusan pada ambang OOF skenario terpilih.')
    st.write('Citra dibaca menjadi tiga kanal, diubah menjadi 224 × 224, dan diproses seperti notebook. Augmentasi tidak digunakan.')
    st.info('Hasil merupakan prediksi model penelitian, bukan diagnosis klinis. Skor probabilitas bukan ukuran kepastian diagnosis.')
    st.caption('Belum dicocokkan dengan prediksi notebook pada subjek nyata. Model berasal dari berkas yang diberikan peneliti.')

with tab_input:
    st.subheader('1. Unggah citra')
    uploads = st.file_uploader('Pilih 12 PNG atau satu ZIP berisi 12 PNG', type=['png','zip'], accept_multiple_files=True)
    if not uploads:
        st.info('Siapkan satu subjek dengan 4 representasi × 3 bidang potong.')
    else:
        payload = [(x.name, x.getvalue()) for x in uploads]
        fingerprint = hashlib.sha256(scenario.encode()+b''.join(n.encode()+d for n,d in payload)).hexdigest()
        try:
            images, subject = collect_images(payload)
        except Exception as error:
            st.error(str(error))
            st.stop()
        st.caption(f'Subjek: {subject} · {len(images)}/12 kombinasi')
        for plane in PLANES:
            with st.expander(plane.capitalize(), expanded=True):
                for column,tissue in zip(st.columns(4), TISSUES):
                    with column:
                        st.caption(tissue)
                        record = images.get((plane,tissue))
                        if record:
                            st.image(Image.open(io.BytesIO(record['data'])), width='stretch')
                        else:
                            st.warning('Belum diunggah')
        missing = EXPECTED-set(images)
        if missing:
            st.warning('Masih kurang: '+', '.join(f'{t} {p}' for p,t in sorted(missing)))
        confirmed = st.checkbox('Saya memastikan seluruh citra berasal dari satu subjek dan pemindaian yang sama.')
        if available < 15:
            st.warning('Lengkapi 15 model skenario terpilih. Periksa folder sumber dan hindari berkas duplikat.')
        st.subheader('2. Jalankan model')
        if st.button(f'Jalankan prediksi {scenario}', type='primary', disabled=bool(missing) or available<15 or not confirmed):
            st.session_state.pop('result',None)
            bar = st.progress(0, text='Memuat model pertama…')
            try:
                result = predict_subject(images, lambda fraction,text: bar.progress(fraction,text=text), scenario=scenario)
                result['subject'] = subject
                st.session_state['result'] = (fingerprint,result)
            except Exception as error:
                st.error(f'Prediksi gagal: {error}')
            finally:
                bar.empty()
        saved = st.session_state.get('result')
        if saved and saved[0] == fingerprint:
            result = saved[1]
            st.subheader('3. Hasil prediksi')
            a,b,c = st.columns(3)
            a.metric('Prediksi model', result['prediction'])
            b.metric('Skor probabilitas ASD', f'{result["prob_asd"]:.4f}')
            c.metric('Ambang keputusan', f'{result["threshold"]:.2f}')
            frame = pd.DataFrame({'Bidang potong':list(result['plane_probabilities']), 'Probabilitas ASD':list(result['plane_probabilities'].values())})
            st.dataframe(frame, hide_index=True, width='stretch')
            st.bar_chart(frame.set_index('Bidang potong'), color='#337a9c')
            with st.expander('Rincian empat representasi dan lima fold'):
                st.dataframe(pd.DataFrame(result['tissue_probabilities']), hide_index=True)
                st.json(result['fold_probabilities'], expanded=False)
            export = pd.DataFrame(result['tissue_probabilities'])
            export['final_prob_asd'] = result['prob_asd']
            export['threshold'] = result['threshold']
            export['prediction'] = result['prediction']
            export['scenario'] = result['scenario']
            st.download_button('Unduh rincian CSV', export.to_csv(index=False).encode(), f'hasil_prediksi_{scenario.lower()}.csv', 'text/csv')
            st.download_button('Unduh hasil lengkap JSON', json.dumps(result,indent=2).encode(), f'hasil_prediksi_{scenario.lower()}.json', 'application/json')
            st.caption('Interpretasikan sebagai keluaran model penelitian. Tidak ada accuracy baru yang dihitung dari satu unggahan.')
