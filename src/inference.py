"""
Submodule for model inference (and other tools).
Usage example:

# model inference
>>> preds, probs = get_predictions(model, imgs, device, temperature=1.0, sample=False)
# otherwise, for single scene:
>>> preds, probs = get_predictions(model, [img], device, temperature=1.0, sample=False)

# cloud removing (composite)
>>> composite_image, source_map, cur_mask = composite(imgs, model, device)
"""



import torch



def _predict_single(model, x, device, temperature=1.0, sample=False):
    """
    Single image prediction with temperature scaling.
    x: (7, H, W)
    """
    logits = model(x.unsqueeze(0).to(device))
    probs  = torch.softmax(logits / temperature, dim=1)

    if sample:
        B, C, H, W = probs.shape
        probs_flat  = probs.permute(0, 2, 3, 1).reshape(-1, C)
        pred        = torch.multinomial(probs_flat, 1).reshape(H, W)
    else:
        pred = probs.argmax(dim=1)[0]

    return pred.cpu(), probs[0].cpu()


def get_predictions(model, imgs, device, temperature=1.0, sample=False):
    """
    Pixelwise predictions for a batch of images.

    temperature > 1 -> softer distribution (less confident)
    temperature < 1 -> harder distribution (more confident)

    sample=True  -> sample classes from probability distribution
    sample=False -> argmax (default)

    imgs: (N, 7, H, W)
    temperature: float, default=1.0
    sample: bool,  default=False

    Returns:
        preds: (N, H, W)
        probs: (N, 4, H, W)
    """
    model.eval()
    preds, probs = [], []

    with torch.no_grad():
        for i in range(len(imgs)):
            pred, prob = _predict_single(model, imgs[i], device, temperature, sample)
            preds.append(pred)
            probs.append(prob)

    return torch.stack(preds), torch.stack(probs)


def composite(imgs, model, device):
    """
    Progressing composing to replace clouds from images.
    For every pixels on the image, try to find next replacements:
    1. thick cloud (1) -> replace with thin cloud (2) from other scenes
    2. thin cloud (2)  -> replace with shadow (3) from other scenes
    3. shadow (3)      -> заменяем на clear (0) from other scenes
    
    imgs: (N, 7, H, W) tensor
    returns: composite_image (7, H, W), source_map (H, W), mask_map (H, W)
    """
    model.eval()
    N, C, H, W = imgs.shape

    preds = []
    with torch.no_grad():
        for i in range(N):
            pred = model(imgs[i].unsqueeze(0).to(device)).argmax(dim=1)[0].cpu()
            preds.append(pred)
    preds = torch.stack(preds)  # (N, H, W)

    composite_image = imgs[0].clone()
    cur_mask = preds[0].clone()
    source_map = torch.zeros(H, W, dtype=torch.long)

    replacement_order = [
        (1, [2, 3, 0]),
        (2, [3, 0]),
        (3, [0]),
    ]

    for bad_class, preferred_classes in replacement_order:
        bad_pixels = (cur_mask == bad_class)  # (H, W) - pixels to remove for curr step (thick -> thin -> shadow)
        if not bad_pixels.any():
            continue
        
        print(f"- Trying to replace {bad_class}: total {bad_pixels.sum().item()} pixels")

        for preferred_class in preferred_classes:
            if not bad_pixels.any():
                break

            # for evere scene
            for i in range(1, N):
                # finding pixels in scene [i] with preferred class (to replace bad pixels)
                good_in_scene = (preds[i] == preferred_class) & bad_pixels

                if good_in_scene.any():
                    # updating pixels
                    composite_image[:, good_in_scene] = imgs[i][:, good_in_scene]
                    cur_mask[good_in_scene] = preferred_class
                    source_map[good_in_scene] = i

                    # updating everything else
                    bad_pixels = (cur_mask == bad_class)

                    if not bad_pixels.any():
                        break

    return composite_image, source_map, cur_mask
