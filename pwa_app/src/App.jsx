import { useState } from 'react'
import SplashScreen from './screens/SplashScreen'
import CameraScreen from './screens/CameraScreen'
import ResultScreen from './screens/ResultScreen'
import './App.css'

export default function App() {
  const [screen, setScreen] = useState('splash')
  const [result, setResult] = useState(null)

  const goTo = (screenName, data = null) => {
    if (data) setResult(data)
    setScreen(screenName)
  }

  return (
    <div className="app-root">
      {screen === 'splash' && (
        <SplashScreen onStart={() => goTo('camera')} />
      )}
      {screen === 'camera' && (
        <CameraScreen
          onResult={(data) => goTo('result', data)}
          onBack={() => goTo('splash')}
        />
      )}
      {screen === 'result' && (
        <ResultScreen
          result={result}
          onScanAgain={() => goTo('camera')}
          onHome={() => goTo('splash')}
        />
      )}
    </div>
  )
}