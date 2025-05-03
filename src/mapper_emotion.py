# =========== Engagement Mapping Function ===========
def map_emotion_to_engagement(emotion_label):
    """
    Maps the detected emotion label to an engagement state.
    
    Args:
        emotion_label (str or None): The dominant emotion detected by DeepFace,
                                     or None/Error if no face/emotion found.
    
    Returns:
        tuple: A tuple containing:
            - str: The engagement state ('Engaged' or 'Not Engaged').
            - str: A reason string providing context (e.g., emotion detected or why not engaged).
    """
    valid_emotions = {'sad', 'angry', 'surprise', 'fear', 'happy', 'neutral', 'disgust'}
    if emotion_label and emotion_label.lower() in valid_emotions:
        return 'Engaged', f"({emotion_label} Detected)"
    elif emotion_label == "Error":
        return 'Not Engaged', "(Analysis Error)"
    else:
        return 'Not Engaged', "(No Face Detected)"