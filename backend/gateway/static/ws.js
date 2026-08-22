class WsClient {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.listeners = [];
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 5000;
    this.reconnectTimer = null;
    this.intentionalClose = false;
  }

  connect() {
    try {
      this.ws = new WebSocket(this.url);
      this.ws.onopen = () => {
        this.reconnectDelay = 1000;
      };
      this.ws.onmessage = (event) => {
        let msg;
        try {
          msg = JSON.parse(event.data);
        } catch {
          msg = { raw: event.data };
        }
        this.listeners.forEach((cb) => cb(msg));
      };
      this.ws.onclose = () => {
        if (!this.intentionalClose) {
          this.scheduleReconnect();
        }
      };
      this.ws.onerror = () => {
        if (!this.intentionalClose) {
          this.scheduleReconnect();
        }
      };
    } catch (err) {
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    if (this.reconnectTimer) return;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
    }, this.reconnectDelay);
  }

  onMessage(callback) {
    this.listeners.push(callback);
  }

  offMessage(callback) {
    this.listeners = this.listeners.filter((cb) => cb !== callback);
  }

  send(msg) {
    const payload = typeof msg === 'string' ? msg : JSON.stringify(msg);
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(payload);
    }
  }

  close() {
    this.intentionalClose = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

export { WsClient };
