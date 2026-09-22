import os
import time
import cv2
import numpy as np
from plate_cleaner import PlateCleaner

def run_tests():
    ref_dir = "Imagens Referência com Placa"
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)

    cleaner = PlateCleaner()

    if os.path.exists(ref_dir):
        files = [(os.path.join(ref_dir, f), f) for f in sorted(os.listdir(ref_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    else:
        # Sample images from Galeria de imagens
        galeria_dir = "Galeria de imagens"
        files = []
        if os.path.exists(galeria_dir):
            subdirs = sorted([d for d in os.listdir(galeria_dir) if os.path.isdir(os.path.join(galeria_dir, d))])
            for d in subdirs[:15]:
                dpath = os.path.join(galeria_dir, d)
                imgs = [f for f in os.listdir(dpath) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if imgs:
                    files.append((os.path.join(dpath, imgs[0]), f"{d}_{imgs[0]}"))
        files = files[:12]

    print(f"Iniciando processamento de {len(files)} imagens de teste...\n")
    start_time = time.time()

    success_count = 0
    for idx, (input_path, filename) in enumerate(files, 1):
        t0 = time.time()
        orig, mask_vis, res, boxes = cleaner.process_image(input_path)
        dt = (time.time() - t0) * 1000

        # Create 3-panel comparison: [Original | Máscara / Detecção | Resultado Final]
        h, w = orig.shape[:2]
        # Overlay mask onto original for clarity
        overlay = orig.copy()
        mask_pixels = np.any(mask_vis > 0, axis=-1)
        overlay[mask_pixels] = cv2.addWeighted(orig, 0.4, mask_vis, 0.6, 0)[mask_pixels]
        for box in boxes:
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Scale down large images for easy side-by-side inspection if needed
        max_h = 720
        if h > max_h:
            scale = max_h / h
            nw, nh = int(w * scale), int(h * scale)
            p1 = cv2.resize(orig, (nw, nh))
            p2 = cv2.resize(overlay, (nw, nh))
            p3 = cv2.resize(res, (nw, nh))
        else:
            p1, p2, p3 = orig, overlay, res

        # Add headers
        for img, text in [(p1, "ORIGINAL"), (p2, "DETECCAO E MASCARA"), (p3, "RESULTADO FINAL")]:
            cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

        comparison = np.hstack([p1, p2, p3])
        out_name = f"resultado_{os.path.splitext(filename)[0]}.jpg"
        out_path = os.path.join(output_dir, out_name)

        # Save with PIL to ensure non-ascii Windows compatibility
        from PIL import Image
        comp_rgb = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
        Image.fromarray(comp_rgb).save(out_path, quality=92)

        status = f"DETECTADO ({len(boxes)} placa(s))" if boxes else "SEM PLACA"
        print(f"[{idx:02d}/{len(files):02d}] {filename} -> {status} em {dt:.1f}ms -> Salvo em {out_name}")
        if boxes:
            success_count += 1

    total_time = time.time() - start_time
    print(f"\n--- Resumo dos Testes ---")
    print(f"Total processadas: {len(files)}")
    print(f"Placas identificadas e limpas: {success_count}/{len(files)}")
    print(f"Tempo total: {total_time:.2f}s (Média: {(total_time/len(files))*1000:.1f}ms por imagem)")
    print(f"Resultados salvos na pasta: {output_dir}")

if __name__ == "__main__":
    run_tests()
