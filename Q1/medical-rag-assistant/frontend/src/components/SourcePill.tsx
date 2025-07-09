
import { Source } from '../types';

interface SourcePillProps {
  source: Source;
}

const SourcePill = ({ source }: SourcePillProps) => {
  return (
    <button 
      className="inline-flex items-center px-3 py-1 text-xs font-medium text-blue-700 bg-blue-50 border border-blue-200 rounded-full hover:bg-blue-100 transition-colors duration-200"
      onClick={() => console.log(`Clicked source: ${source.name}`)}
    >
      {source.name}
    </button>
  );
};

export default SourcePill;
