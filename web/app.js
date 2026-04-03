const STORAGE_KEY = "lm_tts_chats_v1";
const TEMP_KEY = "kokoro_temperature";
const VOICE_KEY = "kokoro_voice";
const MODEL_KEY = "kokoro_model";
const BOT_AVATAR_KEY = "kokoro_bot_avatar";
const USER_AVATAR_KEY = "kokoro_user_avatar";
const HISTORY_LIMIT = 25;

const DEFAULT_VOICE = "af_nicole";
const DEFAULT_SYSTEM_PROMPT =
  "You are a helpful AI assistant who responds concisely and clearly. Keep answers friendly and readable. Make sure your answers are suitable to be read aloud by a text-to-speech engine.";

const DEFAULT_REMINDER_PROMPT = "";
const DEFAULT_REMINDER_THRESHOLD = 2000;

const DEFAULT_BOT_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:65%;height:65%"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>`;
const DEFAULT_USER_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:65%;height:65%"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;

const elements = {
  chatList: document.getElementById("chat-list"),
  newChatBtn: document.getElementById("new-chat-btn"),
  messages: document.getElementById("messages"),
  composer: document.getElementById("composer"),
  promptInput: document.getElementById("prompt-input"),
  redoBtn: document.getElementById("redo-btn"),
  stopAudioBtn: document.getElementById("stop-audio-btn"),
  status: document.getElementById("status"),
  temperatureInput: document.getElementById("temperature-input"),
  modelSelect: document.getElementById("model-select"),
  voiceSelect: document.getElementById("voice-select"),
  // Settings dialog
  settingsBtn: document.getElementById("settings-btn"),
  settingsDialog: document.getElementById("settings-dialog"),
  settingsClose: document.getElementById("settings-close"),
  systemPromptInput: document.getElementById("system-prompt-input"),
  applySystemPromptBtn: document.getElementById("apply-system-prompt"),
  closeSystemPromptBtn: document.getElementById("close-system-prompt"),
  reminderPromptInput: document.getElementById("reminder-prompt-input"),
  reminderThresholdInput: document.getElementById("reminder-threshold-input"),
  ttsToggle: document.getElementById("tts-toggle"),
  // Global Settings dialog
  globalSettingsBtn: document.getElementById("global-settings-btn"),
  globalSettingsDialog: document.getElementById("global-settings-dialog"),
  globalSettingsClose: document.getElementById("global-settings-close"),
  closeGlobalSettingsBtn: document.getElementById("close-global-settings"),
  // Delete chat dialog
  deleteChatDialog: document.getElementById("delete-chat-dialog"),
  deleteChatDialogClose: document.getElementById("delete-chat-dialog-close"),
  closeDeleteChatDialogBtn: document.getElementById("close-delete-chat-dialog"),
  confirmDeleteChatDialogBtn: document.getElementById("confirm-delete-chat-dialog"),
  deleteChatTitle: document.getElementById("delete-chat-title"),
  // Typing indicator
  typingIndicator: document.getElementById("typing-indicator"),
  // Chat header
  chatHeaderName: document.getElementById("chat-header-name"),
  chatHeaderTtsStatus: document.getElementById("chat-header-tts-status"),
  chatHeaderStatus: document.getElementById("chat-header-status"),
  chatHeaderAvatar: document.getElementById("chat-header-avatar"),
  // Avatar uploading
  botAvatarPreview: document.getElementById("bot-avatar-preview"),
  botAvatarInput: document.getElementById("bot-avatar-input"),
  botAvatarRemove: document.getElementById("bot-avatar-remove"),
  userAvatarPreview: document.getElementById("user-avatar-preview"),
  userAvatarInput: document.getElementById("user-avatar-input"),
  userAvatarRemove: document.getElementById("user-avatar-remove"),
  // Sidebar
  sidebar: document.querySelector(".sidebar"),
  appShell: document.querySelector(".app-shell"),
};

const state = {
  chats: [],
  activeChatId: null,

  pending: false,
  temperature: 0.8,
  status: "",
  models: [],
  model: null,
  voices: [],
  voice: null,
  audioPlaying: false,
  editingMessageId: null,
  editingValue: "",
  editingChatId: null,
  editingChatValue: "",
  deleteChatId: null,
};

let audioPlayer = null;

function generateRandomSoftColor() {
  const hue = Math.floor(Math.random() * 360);
  const saturation = 65;
  const lightness = 65;
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

document.addEventListener("DOMContentLoaded", init);

async function init() {
  loadState();
  bindEvents();
  bindSettingsEvents();

  if (!state.chats.length) {
    const chat = createChat();
    state.chats.push(chat);
    state.activeChatId = chat.id;
  } else if (!state.activeChatId && state.chats[0]) {
    state.activeChatId = state.chats[0].id;
  }

  elements.temperatureInput.value = state.temperature.toFixed(1);
  render();

  await refreshModels();
  await refreshVoices();
}

function bindEvents() {
  elements.newChatBtn.addEventListener("click", () => {
    const chat = createChat();
    state.chats.push(chat);
    state.activeChatId = chat.id;
    saveState();
    render();
    elements.promptInput.focus();
  });

  elements.composer.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = elements.promptInput.value.trim();
    if (!text || state.pending) return;
    const chat = getActiveChat();
    if (!chat) return;

    const message = {
      id: newId(),
      role: "user",
      content: text,
      createdAt: new Date().toISOString(),
    };
    chat.messages.push(message);
    updateChatTitle(chat);
    elements.promptInput.value = "";
    autoResizeTextarea(elements.promptInput);
    saveState();
    renderMessages();
    await requestAssistantResponse(chat);
    elements.promptInput.focus();
  });

  elements.redoBtn.addEventListener("click", async () => {
    if (state.pending) return;
    const chat = getActiveChat();
    if (!chat || !chat.messages.length) return;
    const last = chat.messages[chat.messages.length - 1];
    if (last.role !== "assistant") return;
    chat.messages.pop();
    saveState();
    renderMessages();
    await requestAssistantResponse(chat);
  });

  elements.temperatureInput.addEventListener("change", () => {
    const value = clamp(parseFloat(elements.temperatureInput.value) || 0.7, 0, 1);
    state.temperature = value;
    elements.temperatureInput.value = value.toFixed(1);
    localStorage.setItem(TEMP_KEY, String(value));
  });

  elements.modelSelect.addEventListener("change", () => {
    const selection = elements.modelSelect.value;
    state.model = selection || null;
    if (selection) {
      localStorage.setItem(MODEL_KEY, selection);
    } else {
      localStorage.removeItem(MODEL_KEY);
    }
    updateControls();
    updateChatHeader();
  });

  elements.voiceSelect.addEventListener("change", () => {
    const selection = elements.voiceSelect.value;
    state.voice = selection || null;
    if (selection) {
      localStorage.setItem(VOICE_KEY, selection);
    } else {
      localStorage.removeItem(VOICE_KEY);
    }
    updateControls();
    updateChatHeader();
  });

  if (elements.stopAudioBtn) {
    elements.stopAudioBtn.addEventListener("click", () => {
      if (!state.audioPlaying) return;
      stopAudio(true);
    });
  }

  // Auto-resize textarea
  elements.promptInput.addEventListener("input", () => {
    autoResizeTextarea(elements.promptInput);
  });

  elements.promptInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey) return;
    event.preventDefault();
    elements.composer.requestSubmit();
  });
}

function autoResizeTextarea(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 128) + "px";
}



// ---------- Settings dialog (native <dialog>) ----------

function bindSettingsEvents() {
  // --- Chat Settings ---
  elements.settingsBtn.addEventListener("click", () => {
    renderSettingsForm();
    elements.settingsDialog.showModal();
  });

  elements.settingsClose.addEventListener("click", () => {
    elements.settingsDialog.close();
  });

  elements.settingsDialog.addEventListener("click", (event) => {
    if (event.target === elements.settingsDialog) {
      elements.settingsDialog.close();
    }
  });

  // --- Global Settings ---
  elements.globalSettingsBtn.addEventListener("click", () => {
    elements.globalSettingsDialog.showModal();
  });

  elements.globalSettingsClose.addEventListener("click", () => {
    elements.globalSettingsDialog.close();
  });

  elements.closeGlobalSettingsBtn.addEventListener("click", () => {
    elements.globalSettingsDialog.close();
  });

  elements.applySystemPromptBtn.addEventListener("click", () => {
    applySystemPrompt();
    elements.settingsDialog.close();
  });

  elements.closeSystemPromptBtn.addEventListener("click", () => {
    elements.settingsDialog.close();
  });

  elements.ttsToggle.addEventListener("change", () => {
    const chat = getActiveChat();
    if (!chat) return;
    chat.ttsEnabled = elements.ttsToggle.checked;
    saveState();
    updateControls();
  });

  // Avatar Upload Logic
  elements.botAvatarPreview.addEventListener("click", () => {
    elements.botAvatarInput.click();
  });

  elements.botAvatarInput.addEventListener("change", async (e) => {
    if (!e.target.files?.length) return;
    try {
      const b64 = await resizeImage(e.target.files[0], 45);
      const chat = getActiveChat();
      if (!chat) return;
      chat.botAvatar = b64;
      saveState();
      elements.botAvatarPreview.innerHTML = `<img src="${b64}" />`;
      updateChatHeader();
      renderMessages();
    } catch (err) {
      console.error(err);
      setStatus("Failed to load assistant avatar.");
    }
    e.target.value = "";
  });

  elements.botAvatarRemove.addEventListener("click", () => {
    const chat = getActiveChat();
    if (!chat) return;
    delete chat.botAvatar;
    saveState();
    elements.botAvatarPreview.innerHTML = DEFAULT_BOT_SVG;
    updateChatHeader();
    renderMessages();
  });

  elements.userAvatarPreview.addEventListener("click", () => {
    elements.userAvatarInput.click();
  });

  elements.userAvatarInput.addEventListener("change", async (e) => {
    if (!e.target.files?.length) return;
    try {
      const b64 = await resizeImage(e.target.files[0], 45);
      const chat = getActiveChat();
      if (!chat) return;
      chat.userAvatar = b64;
      saveState();
      elements.userAvatarPreview.innerHTML = `<img src="${b64}" />`;
      renderMessages();
    } catch (err) {
      console.error(err);
      setStatus("Failed to load user avatar.");
    }
    e.target.value = "";
  });

  elements.userAvatarRemove.addEventListener("click", () => {
    const chat = getActiveChat();
    if (!chat) return;
    delete chat.userAvatar;
    saveState();
    elements.userAvatarPreview.innerHTML = DEFAULT_USER_SVG;
    renderMessages();
  });

  // --- Remove Chat Dialog ---
  elements.deleteChatDialogClose.addEventListener("click", () => {
    elements.deleteChatDialog.close();
  });

  elements.closeDeleteChatDialogBtn.addEventListener("click", () => {
    elements.deleteChatDialog.close();
  });

  elements.confirmDeleteChatDialogBtn.addEventListener("click", () => {
    deleteChat(state.deleteChatId);
    elements.deleteChatDialog.close();
  })
}

function resizeImage(file, maxSize = 128) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        let { width, height } = img;
        if (width > height) {
          if (width > maxSize) {
            height = Math.round((height * maxSize) / width);
            width = maxSize;
          }
        } else {
          if (height > maxSize) {
            width = Math.round((width * maxSize) / height);
            height = maxSize;
          }
        }
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, width, height);
        resolve(canvas.toDataURL("image/webp", 0.8));
      };
      img.onerror = reject;
      img.src = e.target.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function renderSettingsForm() {
  const chat = getActiveChat();
  if (!chat) return;
  const systemMessage = chat.messages.find((m) => m.role === "system");
  elements.systemPromptInput.value = systemMessage ? systemMessage.content : "";
  elements.reminderPromptInput.value = chat.reminderPrompt || "";
  elements.reminderThresholdInput.value = chat.reminderThreshold ?? DEFAULT_REMINDER_THRESHOLD;
  elements.ttsToggle.checked = chat.ttsEnabled !== false;

  const botAvatar = chat.botAvatar;
  elements.botAvatarPreview.innerHTML = botAvatar ? `<img src="${botAvatar}" />` : DEFAULT_BOT_SVG;

  const userAvatar = chat.userAvatar;
  elements.userAvatarPreview.innerHTML = userAvatar ? `<img src="${userAvatar}" />` : DEFAULT_USER_SVG;
}

function loadState() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    state.chats = stored.chats ?? [];
    state.activeChatId = stored.activeChatId ?? null;

    state.chats.forEach((chat) => {
      if (!chat.color) {
        chat.color = generateRandomSoftColor();
      }
      if (chat.ttsEnabled === undefined) chat.ttsEnabled = true;
      if (chat.reminderPrompt === undefined) chat.reminderPrompt = DEFAULT_REMINDER_PROMPT;
      if (chat.reminderThreshold === undefined) chat.reminderThreshold = DEFAULT_REMINDER_THRESHOLD;
    });
  } catch (error) {
    console.warn("Failed to load chats", error);
    state.chats = [];
  }

  const savedTemp = parseFloat(localStorage.getItem(TEMP_KEY) || "0.8");
  state.temperature = clamp(isNaN(savedTemp) ? 0.8 : savedTemp, 0, 1);
  state.model = localStorage.getItem(MODEL_KEY);
  state.voice = localStorage.getItem(VOICE_KEY);
}

function saveState() {
  const payload = {
    chats: state.chats.map((chat) => ({
      ...chat,
      messages: chat.messages.map(({ audioUrl, ...rest }) => rest),
    })),
    activeChatId: state.activeChatId,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

function createChat() {
  const now = new Date().toISOString();
  return {
    id: newId(),
    title: "New chat",
    color: generateRandomSoftColor(),
    createdAt: now,
    ttsEnabled: true,
    reminderPrompt: DEFAULT_REMINDER_PROMPT,
    reminderThreshold: DEFAULT_REMINDER_THRESHOLD,
    messages: [
      {
        id: newId(),
        role: "system",
        content: DEFAULT_SYSTEM_PROMPT,
        createdAt: now,
      },
    ],
  };
}

function getActiveChat() {
  return state.chats.find((chat) => chat.id === state.activeChatId) || null;
}

function updateChatTitle(chat) {
  if (chat.customTitle) return;

  const firstUser = chat.messages.find((m) => m.role === "user");
  if (firstUser) {
    chat.title = firstUser.content.trim() || "Untitled chat";
  }
}

function render() {
  renderChatList();
  renderMessages();
  renderModelSelect();
  renderVoiceSelect();
  updateStatus();
  updateControls();
  updateChatHeader();
}

function renderChatList() {
  elements.chatList.innerHTML = "";
  if (!state.chats.length) return;

  state.chats.forEach((chat) => {
    const btn = document.createElement("button");
    btn.className = "chat-list__item btn ghost";
    if (chat.id === state.activeChatId) {
      btn.classList.add("active");
    }
    btn.style.setProperty("--chat-color", chat.color);
    btn.textContent = chat.title || "Untitled chat";
    btn.title = chat.title || "Untitled chat";
    btn.addEventListener("click", () => {
      state.activeChatId = chat.id;
      saveState();
      render();
      elements.promptInput.focus();
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "chat-list__delete-btn";
    deleteBtn.innerHTML = "&times;";
    deleteBtn.title = "Delete chat";
    deleteBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      elements.deleteChatTitle.textContent = chat.title;
      state.deleteChatId = chat.id;
      elements.deleteChatDialog.showModal();
    });

    const container = document.createElement("div");
    container.className = "chat-list__item-container";
    if (chat.id === state.activeChatId) {
      container.classList.add("active");
    }
    const isEditing = state.editingChatId === chat.id;

    if (isEditing) {
      const input = document.createElement("input");
      input.type = "text";
      input.className = "chat-list__item-input";
      input.value = state.editingChatValue;
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          saveChatTitle(chat.id, input.value.trim());
        } else if (event.key === "Escape") {
          cancelEditingChat();
        }
      });
      input.addEventListener("blur", () => {
        saveChatTitle(chat.id, input.value.trim());
      });

      const focusInput = () => {
        input.focus();
        input.select();
      };
      if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(focusInput);
      } else {
        setTimeout(focusInput, 0);
      }
      container.appendChild(input);
    } else {
      btn.addEventListener("dblclick", () => {
        startEditingChat(chat.id);
      });
      container.append(btn, deleteBtn);
    }
    elements.chatList.appendChild(container);
  });
}

function startEditingChat(chatId) {
  const chat = state.chats.find((c) => c.id === chatId);
  if (!chat) return;

  state.editingChatId = chatId;
  state.editingChatValue = chat.title || "Untitled chat";
  renderChatList();
}

function cancelEditingChat() {
  state.editingChatId = null;
  state.editingChatValue = "";
  renderChatList();
}

function saveChatTitle(chatId, newTitle) {
  if (state.editingChatId !== chatId) return;

  const chat = state.chats.find((c) => c.id === chatId);
  if (chat && newTitle && newTitle !== chat.title) {
    chat.title = newTitle;
    chat.customTitle = true;
    saveState();
  }

  state.editingChatId = null;
  state.editingChatValue = "";
  renderChatList();

  if (state.activeChatId === chatId) {
    updateChatHeader();
  }
}

function deleteChat(chatId) {
  const index = state.chats.findIndex((c) => c.id === chatId);
  if (index === -1) return;

  state.chats.splice(index, 1);
  state.deleteChatId = null;

  if (state.activeChatId === chatId) {
    state.activeChatId = state.chats[0]?.id || null;
    if (!state.activeChatId) {
      const newChat = createChat();
      state.chats.push(newChat);
      state.activeChatId = newChat.id;
    }
  }

  saveState();
  render();
}

function applySystemPrompt() {
  const chat = getActiveChat();
  if (!chat) return;

  const newPrompt = elements.systemPromptInput.value.trim();

  let systemMessage = chat.messages.find((m) => m.role === "system");
  if (systemMessage) {
    systemMessage.content = newPrompt;
    systemMessage.createdAt = new Date().toISOString();
  } else {
    chat.messages.unshift({
      id: newId(),
      role: "system",
      content: newPrompt,
      createdAt: new Date().toISOString(),
    });
  }

  chat.reminderPrompt = elements.reminderPromptInput.value.trim();
  chat.reminderThreshold = Math.max(0, parseInt(elements.reminderThresholdInput.value, 10) || DEFAULT_REMINDER_THRESHOLD);

  saveState();
  render();
  setStatus("Settings applied.", false);
}

function estimateTokenCount(chat) {
  let totalChars = 0;
  for (const msg of chat.messages) {
    if (msg.role === "user" || msg.role === "assistant") {
      totalChars += (msg.content || "").length;
    }
  }
  return Math.round(totalChars / 4);
}

// ---------- Chat Header ----------

function updateChatHeader() {
  const chat = getActiveChat();
  const title = chat ? (chat.title || "New chat") : "Assistant";
  const modelPart = state.model || "No model";
  const voicePart = state.voice ? ` • ${state.voice}` : "";
  const fullStatus = modelPart + voicePart;
  const truncated = fullStatus.length > 60 ? fullStatus.slice(0, 57) + "…" : fullStatus;
  elements.chatHeaderName.textContent = title;

  if (chat && !chat.ttsEnabled) {
    elements.chatHeaderTtsStatus.classList.remove("hidden");
  } else {
    elements.chatHeaderTtsStatus.classList.add("hidden");
  }

  elements.chatHeaderStatus.textContent = truncated;

  const botAvatar = chat ? chat.botAvatar : null;
  elements.chatHeaderAvatar.innerHTML = botAvatar ? `<img src="${botAvatar}" />` : DEFAULT_BOT_SVG;
}

// ---------- Messages ----------

function renderMessages() {
  const chat = getActiveChat();
  elements.messages.innerHTML = "";
  if (!chat) return;

  chat.messages
    .filter((message) => message.role !== "system")
    .forEach((message) => {
      const isEditing = state.editingMessageId === message.id;
      const isUser = message.role === "user";

      const article = document.createElement("article");
      article.className = `message message--${message.role}`;
      if (isEditing) {
        article.classList.add("message--editing");
      }

      // Avatars
      const botAvatarData = chat.botAvatar;
      const userAvatarData = chat.userAvatar;

      const avatar = document.createElement("div");
      avatar.className = "message__avatar";

      if (isUser) {
        avatar.innerHTML = userAvatarData ? `<img src="${userAvatarData}" />` : DEFAULT_USER_SVG;
      } else {
        avatar.innerHTML = botAvatarData ? `<img src="${botAvatarData}" />` : DEFAULT_BOT_SVG;
      }

      // Bubble
      const bubble = document.createElement("div");
      bubble.className = "message__bubble";

      if (isEditing) {
        // Editor inside bubble
        const editor = document.createElement("textarea");
        const editingValue = state.editingValue ?? "";
        editor.className = "message__editor";
        editor.value = editingValue;
        editor.setAttribute("data-edit-id", message.id);
        editor.rows = Math.max(3, editingValue.split("\n").length + 1);
        editor.addEventListener("input", (event) => {
          state.editingValue = event.target.value;
        });
        editor.addEventListener("keydown", (event) => {
          if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
            event.preventDefault();
            saveEditedMessage();
          } else if (event.key === "Escape") {
            event.preventDefault();
            cancelEditingMessage();
          }
        });

        const actions = document.createElement("div");
        actions.className = "message__editor-actions";

        const cancelBtn = document.createElement("button");
        cancelBtn.type = "button";
        cancelBtn.className = "btn ghost small";
        cancelBtn.textContent = "Cancel";
        cancelBtn.addEventListener("click", cancelEditingMessage);

        const saveBtn = document.createElement("button");
        saveBtn.type = "button";
        saveBtn.className = "btn primary small";
        saveBtn.textContent = "Save";
        saveBtn.addEventListener("click", saveEditedMessage);

        actions.append(cancelBtn, saveBtn);
        bubble.append(editor, actions);
      } else {
        // Content
        const content = document.createElement("div");
        content.className = "message__content message__content--markdown";
        content.innerHTML = parseMarkdown(message.content);
        bubble.appendChild(content);

        // Action buttons (inside bubble, shown on hover)
        const actionsBar = document.createElement("div");
        actionsBar.className = "message__actions";

        if (!isUser) {
          actionsBar.appendChild(createActionBtn("🔊", "Speak", () => speakAssistantMessage(message.id)));
        }
        actionsBar.appendChild(createActionBtn("📋", "Copy", () => copyMessage(message.content)));
        actionsBar.appendChild(createActionBtn("✏️", "Edit", () => startEditingMessage(message.id)));
        actionsBar.appendChild(createActionBtn("🔀", "Branch", () => branchChatFromMessage(message.id)));

        bubble.appendChild(actionsBar);
      }

      article.append(avatar, bubble);
      elements.messages.appendChild(article);
    });

  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function createActionBtn(emoji, title, handler) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "message__action-btn";
  btn.textContent = emoji;
  btn.title = title;
  btn.addEventListener("click", handler);
  return btn;
}

function updateStatus() {
  elements.status.textContent = state.status || "";
}

function updateControls() {
  // Typing indicator
  if (elements.typingIndicator) {
    elements.typingIndicator.classList.toggle("hidden", !state.pending);
  }

  const noModel = !state.model;
  const chat = getActiveChat();
  const ttsEnabled = chat ? chat.ttsEnabled !== false : true;
  const noVoice = !state.voice && ttsEnabled;

  elements.redoBtn.disabled = state.pending || noModel || (ttsEnabled && !state.voice) || !canRedo();

  const disabled = state.pending || noModel || (ttsEnabled && !state.voice);
  elements.promptInput.disabled = disabled;
  const sendBtn = elements.composer.querySelector(".composer__send-btn");
  if (sendBtn) sendBtn.disabled = disabled;

  elements.modelSelect.disabled = !state.models.length;
  elements.voiceSelect.disabled = !state.voices.length;

  if (elements.stopAudioBtn) {
    elements.stopAudioBtn.disabled = !state.audioPlaying;
  }

  // Scroll to bottom when pending starts
  if (state.pending) {
    elements.messages.scrollTop = elements.messages.scrollHeight;
  }
}

function canRedo() {
  const chat = getActiveChat();
  if (!chat || chat.messages.length < 2) return false;
  return chat.messages[chat.messages.length - 1].role === "assistant";
}

async function requestAssistantResponse(chat) {
  if (!chat) return;
  if (!state.model) {
    setStatus("No LM Studio model available.");
    return;
  }
  const ttsEnabled = chat.ttsEnabled !== false;
  if (ttsEnabled && !state.voice) {
    setStatus("No Kokoro voice selected.");
    return;
  }
  const last = chat.messages[chat.messages.length - 1];
  if (!last || last.role !== "user") return;

  setPending(true);
  setStatus("");

  try {
    const systemPromptMessage = chat.messages.find((m) => m.role === "system");
    const otherMessages = chat.messages.filter((m) => m.role !== "system");
    const recentMessages = otherMessages.slice(-HISTORY_LIMIT);

    const formattedMessages = [];
    if (systemPromptMessage) {
      formattedMessages.push({ role: systemPromptMessage.role, content: systemPromptMessage.content });
    }
    for (const m of recentMessages) {
      formattedMessages.push({ role: m.role, content: m.content });
    }

    const payload = {
      messages: formattedMessages,
      temperature: state.temperature,
      model: state.model,
      voice: ttsEnabled ? state.voice : null,
      tts_enabled: ttsEnabled,
    };

    const estimatedTokens = estimateTokenCount(chat);
    if (chat.reminderPrompt && chat.reminderThreshold > 0 && estimatedTokens >= chat.reminderThreshold) {
      payload.reminder_prompt = chat.reminderPrompt;
    }

    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Failed to generate response");
    }

    const assistantMessage = {
      id: newId(),
      role: "assistant",
      content: (data.content || "").trim(),
      createdAt: new Date().toISOString(),
      voice: data.voice || state.voice,
    };
    if (ttsEnabled && data.audio) {
      assistantMessage.audioUrl = `data:audio/wav;base64,${data.audio}`;
    }

    chat.messages.push(assistantMessage);
    updateChatTitle(chat);
    saveState();
    renderMessages();
    renderChatList();
    updateControls();

    if (ttsEnabled && assistantMessage.audioUrl) {
      playAudio(assistantMessage.audioUrl);
    }
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Something went wrong");
  } finally {
    setPending(false);
  }
}

function setPending(flag) {
  state.pending = flag;
  updateControls();
}

async function speakAssistantMessage(messageId) {
  const chat = getActiveChat();
  if (!chat) return;
  const message = chat.messages.find((m) => m.id === messageId);
  if (!message) return;
  const selectedVoice = state.voice;
  if (!selectedVoice) {
    setStatus("No Kokoro voice selected.");
    return;
  }

  try {
    if (message.audioUrl && message.voice === selectedVoice) {
      playAudio(message.audioUrl);
      return;
    }
    setStatus("Generating speech<span class='loader green'></span>", false);
    const audioUrl = await fetchTTS(message.content, selectedVoice);
    message.audioUrl = audioUrl;
    message.voice = selectedVoice;
    saveState();
    playAudio(audioUrl);
  } catch (error) {
    console.error(error);
    setStatus(error.message || "Failed to synthesize audio");
  } finally {
    setTimeout(() => setStatus(""), 1500);
  }
}

async function fetchTTS(text, voice) {
  const response = await fetch("/api/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "TTS error");
  }
  return `data:audio/wav;base64,${data.audio}`;
}

function playAudio(source) {
  stopAudio(false);

  const audio = new Audio(source);
  audioPlayer = audio;
  state.audioPlaying = true;
  updateControls();

  const handleAudioFinished = () => {
    if (audioPlayer === audio) {
      audioPlayer = null;
      state.audioPlaying = false;
      updateControls();
    }
  };

  audio.addEventListener("ended", handleAudioFinished);
  audio.addEventListener("error", (event) => {
    console.error("Audio playback failed", event);
    setStatus("Unable to play audio (check browser permissions).");
    stopAudio(false);
  });

  audio.play().catch((error) => {
    console.error("Audio playback failed", error);
    setStatus("Unable to play audio (check browser permissions).");
    stopAudio(false);
  });
}

function stopAudio(showStatus = false) {
  if (audioPlayer) {
    audioPlayer.pause();
    audioPlayer.currentTime = 0;
    audioPlayer = null;
  }
  if (state.audioPlaying) {
    state.audioPlaying = false;
    updateControls();
  }
  if (showStatus) {
    setStatus("Audio stopped.", false);
  }
}

function copyMessage(text) {
  navigator.clipboard
    .writeText(text)
    .then(() => setStatus("Copied!", false))
    .catch(() => setStatus("Copy failed"));
}

function startEditingMessage(messageId) {
  const chat = getActiveChat();
  if (!chat) return;
  const message = chat.messages.find((m) => m.id === messageId);
  if (!message || message.role === "system") return;

  state.editingMessageId = messageId;
  state.editingValue = message.content;
  renderMessages();

  const focusEditor = () => {
    const textarea = elements.messages.querySelector(
      `textarea[data-edit-id="${messageId}"]`
    );
    if (textarea) {
      textarea.focus();
      const length = textarea.value.length;
      textarea.setSelectionRange(length, length);
    }
  };

  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(focusEditor);
  } else {
    setTimeout(focusEditor, 0);
  }
}

function cancelEditingMessage() {
  state.editingMessageId = null;
  state.editingValue = "";
  renderMessages();
}

function saveEditedMessage() {
  const messageId = state.editingMessageId;
  if (!messageId) return;

  const chat = getActiveChat();
  if (!chat) return;

  const index = chat.messages.findIndex((m) => m.id === messageId);
  if (index === -1) return;

  const message = chat.messages[index];
  const trimmed = state.editingValue.trim();

  if (!trimmed) {
    setStatus("Message cannot be empty.");
    return;
  }

  if (message.content === trimmed) {
    cancelEditingMessage();
    setStatus("No changes made.", false);
    return;
  }

  message.content = trimmed;

  if (message.role === "assistant") {
    delete message.audioUrl;
    saveState();
    cancelEditingMessage();
    setStatus("Assistant message updated.", false);
    return;
  }

  if (message.role === "user") {
    chat.messages = chat.messages.slice(0, index + 1);
    saveState();
    cancelEditingMessage();
    requestAssistantResponse(chat);
  }
}

function branchChatFromMessage(messageId) {
  const chat = getActiveChat();
  if (!chat) return;
  const index = chat.messages.findIndex((m) => m.id === messageId);
  if (index === -1) return;

  const newChat = {
    id: newId(),
    title: chat.title,
    color: generateRandomSoftColor(),
    createdAt: new Date().toISOString(),
    ttsEnabled: chat.ttsEnabled,
    reminderPrompt: chat.reminderPrompt,
    reminderThreshold: chat.reminderThreshold,
    messages: chat.messages.slice(0, index + 1).map((message) => ({ ...message })),
  };

  updateChatTitle(newChat);
  state.chats.unshift(newChat);
  state.activeChatId = newChat.id;
  saveState();
  render();
  if (elements.promptInput) {
    elements.promptInput.focus();
  }
  setStatus("Branched chat created.", false);
}

let statusTimeout = null;

function setStatus(message, isError = true, timeoutMs = 5000) {
  state.status = message;
  elements.status.innerHTML = message || "";

  if (statusTimeout) {
    clearTimeout(statusTimeout);
    statusTimeout = null;
  }

  if (!message) {
    elements.status.style.color = "var(--text-muted)";
    return;
  }

  elements.status.style.color = isError ? "#ea4335" : "var(--accent)";

  // Auto-clear most messages after 5 seconds, unless it contains a loader or timeout is disabled
  if (timeoutMs > 0 && !message.includes("loader")) {
    statusTimeout = setTimeout(() => {
      setStatus("");
    }, timeoutMs);
  }
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function newId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `id-${Math.random().toString(36).slice(2, 10)}`;
}

async function refreshModels() {
  try {
    const response = await fetch("/api/models");
    const data = await response.json();
    state.models = data.models || [];

    if (!state.models.length) {
      state.model = null;
      localStorage.removeItem(MODEL_KEY);
      setStatus("No model is currently loaded in LM Studio.");
    } else {
      if (!state.model || !state.models.includes(state.model)) {
        state.model = state.models[0];
        localStorage.setItem(MODEL_KEY, state.model);
      }
      if (state.status === "No model is currently loaded in LM Studio.") {
        setStatus("");
      }
    }
  } catch (error) {
    console.error(error);
    setStatus("Unable to fetch LM Studio models.");
  } finally {
    renderModelSelect();
    updateControls();
    updateChatHeader();
  }
}

async function refreshVoices() {
  try {
    const response = await fetch("/api/voices");
    const data = await response.json();
    state.voices = data.voices || [];

    if (!state.voices.length) {
      state.voice = null;
      localStorage.removeItem(VOICE_KEY);
      setStatus("No Kokoro voices are available.");
    } else {
      const defaultVoice = data.default;
      const hasCurrent = state.voices.some((voice) => voice.name === state.voice);
      if (!hasCurrent) {
        const fallback =
          state.voices.find((voice) => voice.name === defaultVoice)?.name ||
          state.voices[0].name;
        state.voice = fallback;
        localStorage.setItem(VOICE_KEY, fallback);
      }
      if (
        state.status === "No Kokoro voices are available." ||
        state.status === "Unable to fetch Kokoro voices."
      ) {
        setStatus("");
      }
    }
  } catch (error) {
    console.error(error);
    setStatus("Unable to fetch Kokoro voices.");
  } finally {
    renderVoiceSelect();
    updateControls();
    updateChatHeader();
  }
}

function renderModelSelect() {
  const select = elements.modelSelect;
  if (!select) return;

  select.innerHTML = "";

  if (!state.models.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No models";
    select.appendChild(option);
    select.value = "";
    select.disabled = true;
    return;
  }

  state.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    select.appendChild(option);
  });
  select.disabled = false;
  select.value = state.model || state.models[0];
}

function renderVoiceSelect() {
  const select = elements.voiceSelect;
  if (!select) return;

  select.innerHTML = "";

  if (!state.voices.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No voices";
    select.appendChild(option);
    select.value = "";
    select.disabled = true;
    return;
  }

  const voiceOrder = { female: 0, male: 1, unknown: 2 };
  const sortedVoices = [...state.voices].sort((a, b) => {
    const rankA = voiceOrder[a.gender] ?? 2;
    const rankB = voiceOrder[b.gender] ?? 2;
    if (rankA !== rankB) return rankA - rankB;
    return a.name.localeCompare(b.name);
  });

  sortedVoices.forEach((voice) => {
    const option = document.createElement("option");
    option.value = voice.name;
    const label = voice.gender
      ? `${voice.gender === "male" ? "♂️" : voice.gender === "female" ? "♀️" : "?"} · ${voice.name}`
      : voice.name;
    option.textContent = `${label} (${voice.lang_code})`;
    select.appendChild(option);
  });

  select.disabled = false;
  select.value = state.voice || state.voices[0].name;
}

function parseMarkdown(text) {
  if (!text) return "";

  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");

  html = html.replace(/```(?:[a-z]*\n)?([\s\S]*?)```/gm, '<pre class="code-block"><code>$1</code></pre>');
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
  html = html.replace(/^## (.*$)/gim, "<h2>$1</h2>");
  html = html.replace(/^# (.*$)/gim, "<h1>$1</h1>");
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");

  const paragraphs = html.split(/\n\s*\n/);
  return paragraphs
    .map((p) => {
      p = p.trim();
      if (!p) return "";
      if (/^<(h[1-3]|pre)/i.test(p)) return p;
      return `<p>${p.replace(/\n/g, "<br>")}</p>`;
    })
    .join("");
}
