import { useEffect, useState } from 'react'

// In production set VITE_API_URL to the deployed backend; in dev the Vite proxy handles it.
const API_URL = import.meta.env.VITE_API_URL ?? ''

export default function App() {
  const [status, setStatus] = useState('checking...')

  useEffect(() => {
    fetch(`${API_URL}/api/health`)
      .then((r) => r.json())
      .then((d) => setStatus(d.status))
      .catch(() => setStatus('backend not reachable'))
  }, [])

  return (
    <main>
      <h1>HackTheBurgh</h1>
      <p>Backend: {status}</p>
    </main>
  )
}
