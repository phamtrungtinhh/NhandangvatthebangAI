import "./style.css";

const app = document.querySelector("#app");

app.innerHTML = `
  <main class="container">
    <h1>SmartFocus AI Frontend</h1>
    <p>Frontend tách riêng đã được đưa lên GitHub.</p>

    <section class="card">
      <h2>Kết nối backend</h2>
      <p>Backend hiện tại chạy bằng Streamlit tại cổng <strong>8501</strong>.</p>
      <a class="btn" href="http://localhost:8501" target="_blank" rel="noreferrer">Mở ứng dụng Streamlit</a>
    </section>

    <section class="card">
      <h2>Chạy frontend</h2>
      <pre><code>cd frontend
npm install
npm run dev</code></pre>
    </section>
  </main>
`;